import json
import os

import trimesh


def write_solution_files(placed, config, best_result, elapsed_total, mesh_cache):
    method_tag = "grasp"
    step_tag = f"s{int(config.step * 1000):04d}"
    output_json = os.path.join(config.results_dir, f"solucion_{method_tag}_k{config.buffer_size}_{step_tag}.json")
    meta_dir = os.path.join(config.results_dir, "meta")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(placed, f, indent=4)
    print(f"\nGuardado {len(placed)} objetos en {output_json}")

    meta_path = os.path.join(meta_dir, f"meta_{method_tag}_k{config.buffer_size}_{step_tag}.json")
    cache_stats = best_result.get("placement_cache", {})
    metadata = {
        "dataset": config.dataset,
        "buffer_size": config.buffer_size,
        "step": config.step,
        "sequence_index": config.sequence_index,
        "metaheuristic": "grasp",
        "grasp_iterations": config.grasp_iterations,
        "grasp_iterations_data": best_result.get("grasp_iterations", []),
        "rcl_size": config.rcl_size,
        "max_passes": config.max_passes,
        "tail_repack_size": config.tail_repack_size,
        "tail_repack_attempts": config.tail_repack_attempts,
        "random_seed": config.random_seed,
        "placed": len(placed),
        "fill_percent": best_result["fill_percent"],
        "volume_total": best_result["volume_total"],
        "volume_used": best_result["volume_used"],
        "elapsed_packing": best_result["elapsed"],
        "elapsed_total": elapsed_total,
        "elapsed": elapsed_total,
        "output_json": os.path.relpath(output_json, start=config.results_dir),
        "placement_cache_requests": cache_stats.get("cache_requests", 0),
        "placement_cache_hits": cache_stats.get("cache_hits", 0),
        "placement_cache_misses": cache_stats.get("cache_misses", 0),
    }

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=4)
    print(f"Metadata en {meta_path}")

    volumen_usado_chk = 0.0
    for p in placed:
        m = trimesh.load(os.path.join(config.obj_dir, os.path.normpath(p["id"])), force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            m = m.dump().sum()
        volumen_usado_chk += m.volume

    print(f"\n Objetos colocados: {len(placed)}")
    print(f"Volumen contenedor: {best_result['volume_total']:.6f} m³")
    print(f"Volumen usado:     {volumen_usado_chk:.6f} m³")
    print(f"Porcentaje lleno:  {(volumen_usado_chk / best_result['volume_total']) * 100:.2f}%")

    if cache_stats.get("cache_requests", 0) > 0:
        hit_rate = (
            cache_stats["cache_hits"] / cache_stats["cache_requests"] * 100.0
            if cache_stats["cache_requests"] else 0.0
        )
        print(
            f"Cache placement: {cache_stats['cache_hits']} hits, "
            f"{cache_stats['cache_misses']} misses, "
            f"{cache_stats['cache_requests']} reqs ({hit_rate:.1f}% hit)"
        )
