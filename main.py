import os
import random
import time
from collections import Counter

import numpy as np
import torch

from sequential_packing.cli import parse_args
from sequential_packing.config import build_config
from sequential_packing.mesh import load_mesh_cached, normalize_shape_path
from sequential_packing.output import write_solution_files
from sequential_packing.packing import apply_tail_repack, solve_order


def load_sequence_names(config):
    id2name = torch.load(config.id2name_path, map_location="cpu")
    sequence_data = torch.load(config.sequence_path, map_location="cpu", weights_only=False)
    if isinstance(sequence_data, torch.Tensor):
        sequences = sequence_data.tolist()
    else:
        sequences = sequence_data

    if config.sequence_index < 0 or config.sequence_index >= len(sequences):
        raise IndexError(
            f"sequence-index {config.sequence_index} fuera de rango (total {len(sequences)})"
        )
    sequence = sequences[config.sequence_index]
    return [id2name[int(i)] for i in sequence]


def irregular_key(name, config, mesh_cache):
    path_obj = os.path.join(config.obj_dir, normalize_shape_path(name))
    _, mesh_size, _ = load_mesh_cached(path_obj, mesh_cache)
    return (
        -np.std(mesh_size),
        -np.prod(mesh_size),
        mesh_size[2],
    )


def zhao_key(name, shape_stats):
    stats = shape_stats.get(name)
    if not stats:
        return (0.0, 0.0, 0.0)
    return (-stats["min_dim"], -stats["count"], -stats["max_dim"])


def apply_ordering(names, config, mesh_cache):
    if config.irregular_order:
        return sorted(names, key=lambda name: irregular_key(name, config, mesh_cache))

    if config.regular_order:
        type_counts = Counter(names)
        shape_stats = {}
        for name in type_counts:
            path_obj = os.path.join(config.obj_dir, normalize_shape_path(name))
            _, mesh_size, _ = load_mesh_cached(path_obj, mesh_cache)
            extents = mesh_size
            shape_stats[name] = {
                "min_dim": float(np.min(extents)),
                "max_dim": float(np.max(extents)),
                "count": type_counts[name],
            }
        return sorted(names, key=lambda name: zhao_key(name, shape_stats))

    return names


def main():
    args = parse_args()
    config = build_config(args)
    if config.random_seed is not None:
        random.seed(config.random_seed)

    nombres_shapes = load_sequence_names(config)
    print(
        f"Dataset={config.dataset} | Step={config.step:.3f} m | K={config.buffer_size} | Secuencia={config.sequence_index}"
    )

    mesh_cache = {}
    stable_pose_cache = {}
    ordered_shapes = apply_ordering(nombres_shapes, config, mesh_cache)

    run_start_time = time.perf_counter()
    best_result = solve_order(ordered_shapes, config, mesh_cache, stable_pose_cache)
    best_order = list(ordered_shapes)
    best_result, best_order = apply_tail_repack(best_result, best_order, config, mesh_cache, stable_pose_cache)
    elapsed_total = time.perf_counter() - run_start_time

    write_solution_files(best_result["placements"], config, best_result, elapsed_total, mesh_cache)


if __name__ == "__main__":
    main()
