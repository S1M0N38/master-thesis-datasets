import argparse
import json
import re
import time
from pathlib import Path

import openai

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"
PATH_WRITERS = PATH_ROOT / "writers"


def parse(payload: dict) -> str:
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "stop"
    return choice["message"]["content"]


def clean(desc: str) -> str:
    # Remove newline characters and replace with whitespace
    return re.sub(r"\s*\n\s*", " ", desc)


def main(dataset: str, writer: str) -> None:
    path_classes = PATH_DATASETS / dataset / "classes"
    path_descriptions = PATH_DATASETS / dataset / "descriptions" / writer
    path_embeddings = PATH_DATASETS / dataset / "embeddings" / "ada" / writer
    path_embeddings.mkdir(parents=True, exist_ok=True)

    with open(path_classes / "classes.txt") as f:
        classes = [cls.strip() for cls in f.readlines()]

    for i, cls in enumerate(classes, 1):
        if (path_embeddings / f"{cls}.json").exists():
            print(f"[{i}] Skipping {cls}")
            continue

        with open(path_descriptions / f"{cls}.json") as f:
            description = clean(parse(json.load(f)))

        response = openai.Embedding.create(
            input=description,
            model="text-embedding-ada-002",
        )
        print(f"[{i}] Embedding {cls}...")

        with open(path_embeddings / f"{cls}.json", "w") as f:
            json.dump(response, f, indent=4)
        time.sleep(21)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings from class descriptions.",
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to generate embeddings for",
    )
    parser.add_argument(
        "--writer",
        choices=[w.stem for w in PATH_WRITERS.iterdir() if w.is_file()],
        required=True,
        help="Use descriptions from this writer",
    )

    args = parser.parse_args()

    main(args.dataset, args.writer)
