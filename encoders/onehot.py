import argparse
from pathlib import Path

import numpy as np

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


def main(dataset: str, save: bool = False) -> np.ndarray:
    path_hierarchy = PATH_DATASETS / dataset / "hierarchy"
    path_encodings = PATH_DATASETS / dataset / "encodings"
    path_encodings.mkdir(parents=True, exist_ok=True)

    hierarchy = np.load(path_hierarchy / "hierarchy.npy")
    encodings = np.eye(hierarchy.shape[-1]).astype(np.float32)

    if save:
        np.save(path_encodings / "onehot.npy", encodings)

    return encodings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate One-Hot-Encoding, i.e. identity matrix.",
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to generate encodings for",
    )

    args = parser.parse_args()

    main(args.dataset, save=True)
