import argparse
from pathlib import Path

import numpy as np

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


def lca_to_encodings(lca: np.ndarray) -> np.ndarray:
    num_levels, num_classes = lca.max(), len(lca)
    s = 1 - (lca / num_levels)
    emb = np.zeros((num_classes, num_classes))
    emb[0, 0] = 1.0
    for c in range(1, num_classes):
        emb[c, :c] = np.linalg.solve(emb[:c, :c], s[c, :c])
        emb[c, c] = np.sqrt(1.0 - np.sum(emb[c, :c] ** 2))
    return emb


def main(dataset: str, save: bool = False) -> np.ndarray:
    path_hierarchy = PATH_DATASETS / dataset / "hierarchy"
    path_encodings = PATH_DATASETS / dataset / "encodings"
    path_encodings.mkdir(parents=True, exist_ok=True)

    lca = np.load(path_hierarchy / "lca.npy")
    encodings = lca_to_encodings(lca)

    if save:
        np.save(path_encodings / "barz-denzler.npy", encodings)

    return encodings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate Encodings from hierarchy using "
            "method proposed by Barz and Denzler."
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to generate encodings for",
    )

    args = parser.parse_args()

    main(args.dataset, save=True)
