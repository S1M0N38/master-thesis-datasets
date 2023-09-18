import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"
PATH_WRITERS = PATH_ROOT / "writers"
PATH_EMBEDDERS = PATH_ROOT / "embedders"


def parse(payload: dict) -> list[float]:
    choice = payload["data"][0]
    assert choice["object"] == "embedding"
    return choice["embedding"]


def main(
    dataset: str,
    writer: str,
    embedder: str,
    n_components: int,
    random_state: int = 42,
    save: bool = False,
) -> np.ndarray:
    path_dataset = PATH_DATASETS / dataset
    path_classes = path_dataset / "classes"
    path_embeddings = path_dataset / "embeddings" / embedder / writer
    path_encodings = path_dataset / "encodings" / "desc-pca" / embedder / writer
    path_encodings.mkdir(parents=True, exist_ok=True)

    with open(path_classes / "classes.txt") as f:
        classes = [cls.strip() for cls in f.readlines()]

    embeddings = []
    for cls in classes:
        with open(path_embeddings / f"{cls}.json") as f:
            embeddings.append(parse(json.load(f)))
    embeddings = np.array(embeddings)

    pca = PCA(
        n_components=n_components,
        random_state=random_state,
    )

    encodings = pca.fit_transform(embeddings).astype(np.float32)

    if save:
        np.save(
            path_encodings / f"d{n_components}.npy",
            encodings,
        )

    return encodings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Encoding from embeddings using PCA.",
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to generate encodings for",
    )
    parser.add_argument(
        "--writer",
        choices=[w.stem for w in PATH_WRITERS.iterdir() if w.is_file()],
        required=True,
        help="Use descriptions from this writer",
    )
    parser.add_argument(
        "--embedder",
        choices=[w.stem for w in PATH_EMBEDDERS.iterdir() if w.is_file()],
        required=True,
        help="Use embeddings from this embedder",
    )
    parser.add_argument(
        "--n_components",
        required=True,
        type=int,
        help="Number of components to reduce to",
    )
    parser.add_argument(
        "--random_state",
        default=42,
        type=int,
        help="Random state for t-SNE",
    )

    args = parser.parse_args()

    main(
        args.dataset,
        args.writer,
        args.embedder,
        args.n_components,
        random_state=args.random_state,
        save=True,
    )
