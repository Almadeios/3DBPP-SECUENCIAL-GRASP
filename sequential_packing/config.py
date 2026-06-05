import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PackingConfig:
    dataset: str
    buffer_size: int
    step: float
    sequence_index: int
    restrict_rotations: bool
    tail_repack_size: int
    tail_repack_attempts: int
    random_seed: Optional[int]
    regular_order: bool
    irregular_order: bool
    max_passes: int
    grasp_iterations: int
    grasp_workers: int
    rcl_size: int
    base_dir: str
    obj_dir: str
    sequence_path: str
    id2name_path: str
    results_dir: str
    container_dims: np.ndarray
    eps: float
    concave_ratio_threshold: float
    cavity_margin: float
    cavity_depth_factor: float
    min_support_frac: float
    support_tol: float
    max_positions_per_variant: int
    allow_stable_poses: bool
    rcl_size_effective: int
    nx: int
    ny: int


def build_config(args) -> PackingConfig:
    base_dir = os.path.join("dataset", args.dataset)
    obj_dir = os.path.join(base_dir, "shape_vhacd")
    sequence_path = os.path.join(base_dir, "test_sequence.pt")
    id2name_path = os.path.join(base_dir, "id2shape.pt")
    results_dir = os.path.join("resultados", args.dataset)
    container_dims = np.array([320, 320, 300], dtype=float) / 1000.0
    eps = 1e-6
    container_step = args.step
    nx = max(1, int(np.floor(container_dims[0] / container_step)) + 1)
    ny = max(1, int(np.floor(container_dims[1] / container_step)) + 1)
    return PackingConfig(
        dataset=args.dataset,
        buffer_size=args.buffer_size,
        step=container_step,
        sequence_index=args.sequence_index,
        restrict_rotations=args.restrict_rotations,
        tail_repack_size=args.tail_repack_size,
        tail_repack_attempts=args.tail_repack_attempts,
        random_seed=args.random_seed,
        regular_order=args.regular_order,
        irregular_order=args.irregular_order,
        max_passes=args.max_passes,
        grasp_iterations=args.grasp_iterations,
        rcl_size=args.rcl_size,
        base_dir=base_dir,
        obj_dir=obj_dir,
        sequence_path=sequence_path,
        id2name_path=id2name_path,
        results_dir=results_dir,
        container_dims=container_dims,
        eps=eps,
        concave_ratio_threshold=0.55,
        cavity_margin=0.01,
        cavity_depth_factor=0.25,
        min_support_frac=0.30,
        support_tol=0.5 * container_step,
        max_positions_per_variant=40,
        allow_stable_poses=not args.restrict_rotations,
        grasp_workers=(
            args.grasp_workers
            if args.grasp_workers > 0
            else max(1, min(os.cpu_count() or 1, args.grasp_iterations))
        ),
        rcl_size_effective=max(1, args.rcl_size),
        nx=nx,
        ny=ny,
    )
