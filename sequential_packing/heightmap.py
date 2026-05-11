import numpy as np

from .mesh import aabb_inside_container, get_oriented_variants


def world_to_idx(x, y, config):
    ix = int(np.floor(x / config.step))
    iy = int(np.floor(y / config.step))
    ix = max(0, min(config.nx - 1, ix))
    iy = max(0, min(config.ny - 1, iy))
    return ix, iy


def rect_to_idrange(xmin, xmax, ymin, ymax, config):
    ix0, iy0 = world_to_idx(max(0.0, xmin + config.eps), max(0.0, ymin + config.eps), config)
    ix1, iy1 = world_to_idx(max(0.0, xmax - config.eps), max(0.0, ymax - config.eps), config)
    ix0, ix1 = min(ix0, ix1), max(ix0, ix1)
    iy0, iy1 = min(iy0, iy1), max(iy0, iy1)
    return ix0, ix1, iy0, iy1


def compute_lz_and_support(heightmap, xmin, xmax, ymin, ymax, config):
    ix0, ix1, iy0, iy1 = rect_to_idrange(xmin, xmax, ymin, ymax, config)
    patch = heightmap[ix0:ix1 + 1, iy0:iy1 + 1]
    if patch.size == 0:
        return 0.0, 0.0, 0.0, (ix0, ix1, iy0, iy1)

    lz = float(np.percentile(patch, 90.0))
    support_mask = np.abs(patch - lz) <= config.support_tol
    support_frac = float(np.mean(support_mask))
    return lz, support_frac, patch, (ix0, ix1, iy0, iy1)


def candidate_score(delta_h, patch, lz, translation, size, config):
    if isinstance(patch, float):
        rough = 0.0
    else:
        rough = float(np.mean(np.abs(lz - patch)))

    if isinstance(patch, float):
        frag_penalty = 0.0
    else:
        frag_penalty = float(np.std(patch))

    lam = 0.01
    wall_gaps = [
        translation[0],
        config.container_dims[0] - (translation[0] + size[0]),
        translation[1],
        config.container_dims[1] - (translation[1] + size[1]),
    ]
    wall_gap = max(0.0, min(wall_gaps))

    return (
        delta_h
        + lam * rough
        + 2.0 * frag_penalty,
        translation[2],
        translation[1],
        translation[0],
        wall_gap,
    )


def is_concave_container(volume_real, size, config):
    if size[2] <= 0 or np.prod(size) <= 0:
        return False
    ratio = volume_real / float(np.prod(size))
    return ratio < config.concave_ratio_threshold and size[2] > 0.05


def carve_cavity(heightmap, translation, size, config):
    inner_min_x = translation[0] + config.cavity_margin
    inner_max_x = translation[0] + size[0] - config.cavity_margin
    inner_min_y = translation[1] + config.cavity_margin
    inner_max_y = translation[1] + size[1] - config.cavity_margin
    if inner_min_x >= inner_max_x - config.eps or inner_min_y >= inner_max_y - config.eps:
        return

    ix0, ix1, iy0, iy1 = rect_to_idrange(inner_min_x, inner_max_x, inner_min_y, inner_max_y, config)
    cavity_height = translation[2] + min(size[2] * config.cavity_depth_factor, 0.04)
    heightmap[ix0:ix1 + 1, iy0:iy1 + 1] = np.minimum(heightmap[ix0:ix1 + 1, iy0:iy1 + 1], cavity_height)


def best_pos_with_drop_for_variant(variant, scene, current_height, heightmap, config):
    mesh_oriented = variant["mesh"]
    offset = variant["offset"]
    size = variant["size"]
    bounds = variant["bounds"]
    T_orient = variant["T_orient"]

    best = None
    found = 0
    stop_search = False
    limits = np.maximum(config.container_dims - size, 0.0)

    def make_candidate(translation, score, idx_data):
        tf = np.eye(4)
        tf[:3, 3] = translation
        return {
            "translation": translation.copy(),
            "score": score,
            "variant": variant,
            "used_size": size.copy(),
            "idxs": idx_data,
            "T_world": tf @ T_orient,
        }

    for y in np.arange(0, limits[1] + 1e-6, config.step):
        for x in np.arange(0, limits[0] + 1e-6, config.step):
            xmin = x
            xmax = x + size[0]
            ymin = y
            ymax = y + size[1]
            lz, support_frac, patch, idx_data = compute_lz_and_support(heightmap, xmin, xmax, ymin, ymax, config)
            if support_frac < config.min_support_frac:
                continue

            z = lz + config.eps
            translation = np.array([x, y, z]) - offset
            if not aabb_inside_container(bounds, translation, config.container_dims, config.eps):
                continue

            tf = np.eye(4)
            tf[:3, 3] = translation
            if scene.in_collision_single(mesh_oriented, transform=tf):
                continue

            top_z = float(bounds[1][2] + translation[2])
            delta_h = max(0.0, top_z - current_height)
            sc = candidate_score(delta_h, patch, lz, translation, size, config)
            cand = make_candidate(translation, sc, idx_data)
            if best is None or sc < best["score"]:
                best = cand
            found += 1
            if found >= config.max_positions_per_variant:
                stop_search = True
                break
        if stop_search:
            break

    if best is not None:
        base_translation = best["translation"].copy()
        base_sc = best["score"]
        base_delta_h = max(0.0, float(bounds[1][2] + base_translation[2]) - current_height)

        x0 = base_translation[0] + offset[0]
        y0 = base_translation[1] + offset[1]
        offsets = [config.step, -config.step, 2 * config.step, -2 * config.step]
        neighbors = [
            (x0 + dx, y0 + dy)
            for dx in offsets
            for dy in offsets
            if not (dx == 0 and dy == 0)
        ]

        for xn, yn in neighbors:
            if xn < 0 or yn < 0 or xn > (config.container_dims[0] - size[0] + 1e-9) or yn > (config.container_dims[1] - size[1] + 1e-9):
                continue

            xmin = xn
            xmax = xn + size[0]
            ymin = yn
            ymax = yn + size[1]
            lz, support_frac, patch, idx_data = compute_lz_and_support(heightmap, xmin, xmax, ymin, ymax, config)
            if support_frac < config.min_support_frac:
                continue

            zn = lz + config.eps
            translation = np.array([xn, yn, zn]) - offset
            if not aabb_inside_container(bounds, translation, config.container_dims, config.eps):
                continue

            tf = np.eye(4)
            tf[:3, 3] = translation
            if scene.in_collision_single(mesh_oriented, transform=tf):
                continue

            top_z_n = float(bounds[1][2] + translation[2])
            delta_h_n = max(0.0, top_z_n - current_height)
            if delta_h_n > base_delta_h + 1e-12:
                continue

            sc_n = candidate_score(delta_h_n, patch, lz, translation, size, config)
            if sc_n < base_sc:
                best = make_candidate(translation, sc_n, idx_data)
                base_sc = sc_n
                base_delta_h = delta_h_n

    return best


def best_feasible_position_with_drop_and_stableposes(item, scene, current_height, heightmap, stable_pose_cache, config):
    best_global = None
    for variant in get_oriented_variants(item["path"], item["mesh"], stable_pose_cache, config.allow_stable_poses, max_poses=4):
        cand = best_pos_with_drop_for_variant(variant, scene, current_height, heightmap, config)
        if cand is not None and (best_global is None or cand["score"] < best_global["score"]):
            best_global = cand
    return best_global
