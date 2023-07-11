import argparse
from pathlib import Path

import numpy as np

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


def softmax(x: np.ndarray):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def lca_to_encodings(lca: np.ndarray, beta: float = 0) -> np.ndarray:
    x = -beta * (lca / lca.max())
    return softmax(x)


def main(dataset: str, beta: float, save: bool = False) -> np.ndarray:
    path_hierarchy = PATH_DATASETS / dataset / "hierarchy"
    path_encodings = PATH_DATASETS / dataset / "encodings" / "mbm"
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
            "method proposed by Bertinetto et al. (2020): "
            "Making Better Mistakes: Leveraging Class Hierarchies "
            "with Deep Networks"
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
        help="The amount of One-Hot-Encoding from [0, +inf)",
    )

    args = parser.parse_args()

    main(args.dataset, args.beta, save=True)
