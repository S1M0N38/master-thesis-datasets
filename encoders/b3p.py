import argparse
from pathlib import Path

import numpy as np

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


def lca_to_encodings(lca: np.ndarray, beta: float = 0) -> np.ndarray:
    encodings = 1 - lca / lca.max()
    encodings = np.clip(encodings, a_min=0, a_max=None)
    encodings /= np.sum(encodings, axis=1, keepdims=True)
    return beta * np.eye(len(lca)) + (1 - beta) * encodings


def main(dataset: str, beta: float, save: bool = False) -> np.ndarray:
    path_hierarchy = PATH_DATASETS / dataset / "hierarchy"
    path_encodings = PATH_DATASETS / dataset / "encodings" / "b3p"
    path_encodings.mkdir(parents=True, exist_ok=True)

    lca = np.load(path_hierarchy / "lca.npy")
    encodings = lca_to_encodings(lca, beta)

    if save:
        np.save(path_encodings / f"beta{beta}.npy", encodings)

    return encodings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate Encodings from hierarchy using "
            "method proposed by Perotti, Bertolotto, Pastor & Panisson "
            "in 'Beyond One-Hot-Encoding: Injecting Semantics to Drive Image "
            "Classifiers'."
        ),
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to generate encodings for",
    )

    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="The amount of One-Hot-Encoding from [0, 1)",
    )

    args = parser.parse_args()

    main(args.dataset, args.beta, save=True)
