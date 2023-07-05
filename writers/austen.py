import argparse
import json
import time
from pathlib import Path

import openai

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


def prompt(cls: str) -> str:
    return (
        "You are an helpful assistant that have to provide the description of "
        f"a '{cls}'.\n"
        f"- What a '{cls}' is.\n"
        f"- What a '{cls}' look like (for example color, texture, shape, ...).\n"
        f"- In what context '{cls}' is used or it can be find.\n"
        f"Foucus of visual characteristics of a '{cls}'.\n"
        f"Write 7 short sentences to describe a '{cls}' in encyclopedic style.\n"
    )


def describe(cls: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that provide "
                "descriptions of visual characteristics of words."
            ),
        },
        {"role": "user", "content": prompt(cls)},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    assert isinstance(response, dict)
    return response


def main(dataset: str) -> None:
    path_classes = PATH_DATASETS / dataset / "classes"
    path_descriptions = PATH_DATASETS / dataset / "descriptions" / "austen"
    path_descriptions.mkdir(parents=True, exist_ok=True)

    with open(path_classes / "classes.txt") as f:
        classes = [cls.strip() for cls in f.readlines()]

    for i, cls in enumerate(classes, 1):
        if (path_descriptions / f"{cls}.json").exists():
            print(f"[{i}] Skipping {cls}")
            continue

        response = describe(cls)
        print(f"[{i}] Describing {cls}...")

        with open(path_descriptions / f"{cls}.json", "w") as f:
            json.dump(response, f, indent=4)
        time.sleep(21)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate descriptions from class names.",
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to generate descriptions for",
    )

    args = parser.parse_args()

    main(args.dataset)
