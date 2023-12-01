import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
from tqdm import tqdm
from umap import UMAP

PATH_ROOT = Path(__file__).parent.parent
PATH_DATASETS = PATH_ROOT / "datasets"


def main(
    dataset: str,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    regex: str = "*.npy",
    init: np.ndarray | str = "spectral",
    verbose: bool = False,
    random_state: int = 42,
    n_jobs: int = 1,
):
    path_dataset = PATH_DATASETS / dataset
    path_encodings = path_dataset / "encodings"
    path_projections = path_dataset / "projections" / "umap"
    path_projections.mkdir(parents=True, exist_ok=True)

    encodings = list((path_dataset / "encodings").rglob(regex))
    # pprint(encodings)
    # input(f"Found {len(encodings)} encodings, press enter to continue...")

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=2,
        metric="cosine",
        init=init,  # type: ignore
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    for path_encoding in tqdm(encodings):
        # Generate projection
        encoding = np.load(path_encoding)
        projection = reducer.fit_transform(encoding)
        projection_path = path_projections / path_encoding.relative_to(path_encodings)
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(projection_path, projection)  # type: ignore
        # Plot projection
        # plot(projection, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project Encodings to 2D using UMAP",
    )

    parser.add_argument(
        "--dataset",
        choices=[d.stem for d in PATH_DATASETS.iterdir() if d.is_dir()],
        required=True,
        help="Dataset to project",
    )
    parser.add_argument(
        "--n_neighbors",
        required=True,
        type=int,
        help="Number of neighbors for UMAP",
    )
    parser.add_argument(
        "--min_dist",
        required=True,
        type=float,
        help="Minium distance for UMAP",
    )
    parser.add_argument(
        "--spread",
        default=1.0,
        type=float,
        help="Spread param for UMAP",
    )
    parser.add_argument(
        "--regex",
        default="*.npy",
        type=str,
        help="Regex matching encodings name",
    )
    parser.add_argument(
        "--init",
        type=Path,
        help="Path to initial embedding for UMAP",
    )
    parser.add_argument(
        "--random_state",
        default=42,
        type=int,
        help="Random state for UMAP",
    )
    parser.add_argument(
        "--n_jobs",
        default=1,
        type=int,
        help="Number of jobs for UMAP",
    )

    args = parser.parse_args()

    if args.init is not None:
        args.init = np.load(args.init)
    else:
        args.init = "spectral"

    main(
        args.dataset,
        args.n_neighbors,
        args.min_dist,
        args.spread,
        regex=args.regex,
        init=args.init,
        random_state=args.random_state,
        verbose=False,
        n_jobs=args.n_jobs,
    )
