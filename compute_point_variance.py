#!/usr/bin/env python3
"""
Utility script to compute the mean nearest neighbor distance of the Gaussian
point cloud produced by a trained scene.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
from plyfile import PlyData

from utils.system_utils import searchForMaxIteration

try:
    from scipy.spatial import cKDTree  # type: ignore
except ImportError:
    cKDTree = None

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except ImportError:
    NearestNeighbors = None


def load_xyz(ply_path: str) -> np.ndarray:
    """Load xyz coordinates from a ply file."""
    plydata = PlyData.read(ply_path)
    xyz = np.stack(
        (
            np.asarray(plydata["vertex"]["x"]),
            np.asarray(plydata["vertex"]["y"]),
            np.asarray(plydata["vertex"]["z"]),
        ),
        axis=1,
    )
    return xyz


def resolve_ply_path(model_path: str, iteration: int) -> Tuple[str, int]:
    """Resolve the ply file path for a given model directory and iteration."""
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(point_cloud_dir):
        raise FileNotFoundError(f"Could not find point_cloud directory at {point_cloud_dir}")

    if iteration < 0:
        iteration = searchForMaxIteration(point_cloud_dir)

    ply_path = os.path.join(point_cloud_dir, f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Could not find ply file at {ply_path}")
    return ply_path, iteration


def compute_mean_nn_distance(xyz: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute the mean nearest neighbor distance."""
    if xyz.shape[0] < 2:
        raise ValueError("At least two points are required to compute nearest neighbor distances.")

    if cKDTree is not None:
        tree = cKDTree(xyz)
        distances, _ = tree.query(xyz, k=2)
        per_point = distances[:, 1]
    elif NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
        nn.fit(xyz)
        distances, _ = nn.kneighbors(xyz, n_neighbors=2, return_distance=True)
        per_point = distances[:, 1]
    else:
        raise ImportError(
            "Either SciPy (preferred) or scikit-learn is required for nearest neighbor queries."
        )

    return float(per_point.mean()), per_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the mean nearest neighbor distance of a trained Gaussian Splatting scene."
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Path to the trained scene directory (e.g., output/scene_name).",
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=-1,
        help="Iteration to load. Use -1 (default) to automatically select the latest iteration.",
    )
    parser.add_argument(
        "--point_cloud",
        "-p",
        type=str,
        help="Direct path to a point_cloud.ply file. Overrides --model_path/--iteration when provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.point_cloud and not args.model_path:
        print("You must provide either --model_path or --point_cloud.", file=sys.stderr)
        sys.exit(1)

    if args.point_cloud:
        ply_path = args.point_cloud
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"Could not find ply file at {ply_path}")
        iteration: Optional[int] = None
    else:
        ply_path, iteration = resolve_ply_path(args.model_path, args.iteration)

    xyz = load_xyz(ply_path)
    mean_distance, per_point = compute_mean_nn_distance(xyz)

    print(f"Loaded {xyz.shape[0]} points from {ply_path}")
    if iteration is not None:
        print(f"Iteration: {iteration}")
    print(f"Mean nearest neighbor distance: {mean_distance:.6f}")
    print(f"Median: {np.median(per_point):.6f}, Min: {per_point.min():.6f}, Max: {per_point.max():.6f}")


if __name__ == "__main__":
    main()
