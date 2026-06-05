import os
import random
import time
import concurrent.futures

import numpy as np
from trimesh.collision import CollisionManager

from .heightmap import best_feasible_position_with_drop_and_stableposes, carve_cavity, is_concave_container
from .mesh import load_mesh_cached, normalize_shape_path
from .progress import SilentProgress, tqdm


def run_packing(sequence_ids, config, mesh_cache, stable_pose_cache, show_progress=True, initial_state=None, rng=None, include_state=True):
    rng = rng or random
    if initial_state is not None:
        placed = list(initial_state.get("placements", []))
        scene = initial_state["scene"]
        heightmap = initial_state["heightmap"]
        volumen_usado = float(initial_state.get("volumen_usado", 0.0))
        current_height = float(initial_state.get("current_height", 0.0))
        volumen_total = float(initial_state.get("volumen_total", np.prod(config.container_dims)))
    else:
        placed = []
        scene = CollisionManager()
        heightmap = np.zeros((config.nx, config.ny), dtype=np.float64)
        volumen_total = np.prod(config.container_dims)
        volumen_usado = 0.0
        current_height = 0.0

    buffer = []
    idx = 0
    total_items = len(sequence_ids)
    pbar = tqdm(total=total_items, desc="Empaquetando objetos", ncols=90) if show_progress else SilentProgress()

    start_time = time.perf_counter()
    while idx < total_items or buffer:
        while idx < total_items and len(buffer) < config.buffer_size:
            name = sequence_ids[idx]
            path_obj = os.path.join(config.obj_dir, normalize_shape_path(name))
            mesh, mesh_size, mesh_offset = load_mesh_cached(path_obj, mesh_cache)
            buffer.append({
                "name": name,
                "path": path_obj,
                "mesh": mesh,
                "mesh_size": mesh_size,
                "mesh_offset": mesh_offset,
                "fails": 0,
            })
            idx += 1

        if not buffer:
            break

        candidate_entries = []
        buffer_sorted = sorted(
            enumerate(buffer),
            key=lambda x: -np.prod(x[1]["mesh_size"]),
        )

        for j, item in buffer_sorted:
            cand = best_feasible_position_with_drop_and_stableposes(item, scene, current_height, heightmap, stable_pose_cache, config)
            if cand is not None:
                base_score = cand["score"][0] if isinstance(cand["score"], tuple) else cand["score"]
                candidate_entries.append((base_score, rng.random(), j, cand))

        if not candidate_entries:
            item = buffer.pop(0)
            item["fails"] += 1
            if item["fails"] < 3:
                buffer.append(item)
            continue

        if config.grasp_iterations and config.rcl_size_effective > 0:
            candidate_entries.sort(key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0])
            alpha = rng.uniform(0.1, 0.5)
            scores = [c[0][0] if isinstance(c[0], tuple) else c[0] for c in candidate_entries]
            s_min = scores[0]
            s_max = scores[-1]
            threshold = s_min + alpha * (s_max - s_min)
            rcl = [c for c in candidate_entries if (c[0][0] if isinstance(c[0], tuple) else c[0]) <= threshold]
            if not rcl:
                rcl = candidate_entries[: config.rcl_size_effective]
            chosen = rng.choice(rcl)
            _, _, j, best_data = chosen
        else:
            candidate_entries.sort(key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0])
            _, _, j, best_data = candidate_entries[0]

        item = buffer.pop(j)
        unique_key = f"{item['name']}#{len(placed)}"
        variant = best_data["variant"]
        translation = best_data["translation"]
        used_size = best_data["used_size"]
        idr = best_data["idxs"]
        tf_local = np.eye(4)
        tf_local[:3, 3] = translation
        scene.add_object(unique_key, variant["mesh"], transform=tf_local)

        placed.append({
            "id": item["name"],
            "position_m": list(np.round(translation, 4)),
            "size_m": list(np.round(used_size, 4)),
            "transform_matrix": np.round(best_data["T_world"], 6).tolist(),
            "T_world": np.round(best_data["T_world"], 6).tolist(),
        })

        volumen_usado += item["mesh"].volume
        bounds = variant["bounds"]
        top_z = float(bounds[1][2] + translation[2])
        current_height = max(current_height, top_z)

        ix0, ix1, iy0, iy1 = idr
        heightmap[ix0:ix1 + 1, iy0:iy1 + 1] = np.maximum(heightmap[ix0:ix1 + 1, iy0:iy1 + 1], top_z)
        if is_concave_container(item["mesh"].volume, used_size, config):
            carve_cavity(heightmap, translation, used_size, config)

        pct = (volumen_usado / volumen_total * 100.0) if volumen_total > 0 else 0.0
        pbar.update(1)
        pbar.set_postfix(vol=f"{pct:5.2f}%", h=f"{current_height:.3f}m")

    pbar.close()
    remaining_sequence = [item["name"] for item in buffer] + sequence_ids[idx:]
    elapsed = time.perf_counter() - start_time
    fill_percent = (volumen_usado / volumen_total * 100.0) if volumen_total > 0 else 0.0
    result = {
        "placements": placed,
        "volume_total": volumen_total,
        "volume_used": volumen_usado,
        "fill_percent": fill_percent,
        "placed_count": len(placed),
        "elapsed": elapsed,
        "order": list(sequence_ids),
        "remaining": remaining_sequence,
    }
    if include_state:
        result["state"] = {
            "placements": placed,
            "scene": scene,
            "heightmap": heightmap,
            "volumen_usado": volumen_usado,
            "current_height": current_height,
            "volumen_total": volumen_total,
        }
    return result


def better_result(candidate, current):
    if current is None:
        return True

    max_objects = max(candidate["placed_count"], current["placed_count"], 1)
    obj_score_cand = candidate["placed_count"] / max_objects
    obj_score_curr = current["placed_count"] / max_objects
    vol_score_cand = candidate["fill_percent"] / 100.0
    vol_score_curr = current["fill_percent"] / 100.0
    # Prefer fill_percent slightly more than number of pieces (but not by much)
    alpha = 0.45
    beta = 0.55
    score_cand = alpha * obj_score_cand + beta * vol_score_cand
    score_curr = alpha * obj_score_curr + beta * vol_score_curr
    return score_cand > score_curr


def run_with_passes(order, config, mesh_cache, stable_pose_cache, show_first_pass=True, rng=None, include_state=True):
    remaining_sequence = list(order)
    state = None
    best_result_local = None
    for pass_idx in range(max(1, config.max_passes)):
        if not remaining_sequence:
            break
        show = show_first_pass and pass_idx == 0
        result = run_packing(
            remaining_sequence,
            config,
            mesh_cache,
            stable_pose_cache,
            show_progress=show,
            initial_state=state,
            rng=rng,
            include_state=include_state,
        )
        best_result_local = result
        state = result.get("state")
        remaining_sequence = result["remaining"]
        if not remaining_sequence:
            break
    return best_result_local


def solve_order_worker(order, config, iter_idx, seed):
    rng = random.Random(seed)
    candidate = run_with_passes(
        order,
        config,
        {},
        {},
        show_first_pass=False,
        rng=rng,
        include_state=False,
    )
    return iter_idx, candidate


def solve_order(order, config, mesh_cache, stable_pose_cache, show_first_pass=True):
    best_result_local = None
    iterations = max(1, config.grasp_iterations)
    grasp_iterations = []

    if config.grasp_workers > 1 and iterations > 1:
        root_rng = random.Random(config.random_seed)
        seeds = [root_rng.randint(0, 2**32 - 1) for _ in range(iterations)]
        results = [None] * iterations

        with concurrent.futures.ProcessPoolExecutor(max_workers=config.grasp_workers) as executor:
            futures = {
                executor.submit(solve_order_worker, order, config, iter_idx, seeds[iter_idx]): iter_idx
                for iter_idx in range(iterations)
            }
            for future in concurrent.futures.as_completed(futures):
                iter_idx, candidate = future.result()
                results[iter_idx] = candidate

        for iter_idx, candidate in enumerate(results):
            iter_elapsed = candidate["elapsed"]
            grasp_iterations.append([
                iter_elapsed,
                candidate["placed_count"],
                candidate["fill_percent"],
            ])
            print(
                f"GRASP iter {iter_idx + 1}/{iterations}: "
                f"{candidate['placed_count']} piezas, {candidate['fill_percent']:.2f}%"
            )
            if better_result(candidate, best_result_local):
                best_result_local = candidate
    else:
        rng = random.Random(config.random_seed)
        for iter_idx in range(iterations):
            iter_start = time.perf_counter()
            show = show_first_pass and iter_idx == 0
            candidate = run_with_passes(
                order,
                config,
                mesh_cache,
                stable_pose_cache,
                show_first_pass=show,
                rng=rng,
            )
            iter_elapsed = time.perf_counter() - iter_start
            grasp_iterations.append([
                iter_elapsed,
                candidate["placed_count"],
                candidate["fill_percent"],
            ])
            print(
                f"GRASP iter {iter_idx + 1}/{iterations}: "
                f"{candidate['placed_count']} piezas, {candidate['fill_percent']:.2f}%"
            )
            if better_result(candidate, best_result_local):
                best_result_local = candidate

    if best_result_local is not None:
        best_result_local["grasp_iterations"] = grasp_iterations
    return best_result_local


def apply_tail_repack(base_result, base_order, config, mesh_cache, stable_pose_cache):
    if (
        config.max_passes != 1
        or config.tail_repack_size <= 0
        or len(base_order) <= config.tail_repack_size
    ):
        return base_result, list(base_order)

    tail_size = min(config.tail_repack_size, len(base_order) - 1)
    max_attempts = min(len(base_order) - tail_size + 1, max(1, config.tail_repack_attempts))
    best_res = base_result
    best_ord = list(base_order)
    for attempt in range(max_attempts):
        start_tail = max(0, len(base_order) - tail_size - attempt)
        block = base_order[start_tail:start_tail + tail_size]
        remaining = base_order[:start_tail] + base_order[start_tail + tail_size:]
        if not block or not remaining:
            continue
        new_order = block + remaining
        candidate = solve_order(new_order, config, mesh_cache, stable_pose_cache, show_first_pass=False)
        if better_result(candidate, best_res):
            print(
                f"Tail repack intento {attempt + 1}: {candidate['placed_count']} piezas, "
                f"{candidate['fill_percent']:.2f}% lleno"
            )
            best_res = candidate
            best_ord = new_order
    return best_res, best_ord
