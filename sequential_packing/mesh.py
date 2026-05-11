import os

import numpy as np
import trimesh
from trimesh.collision import CollisionManager


def normalize_shape_path(name):
    return os.path.normpath(name)


def load_mesh_cached(path, cache):
    if path in cache:
        return cache[path]

    m = trimesh.load(path, force='mesh')
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump().sum()

    mesh = m
    mesh_size = mesh.extents.copy()
    mesh_offset = mesh.bounds[0].copy()
    cache[path] = (mesh, mesh_size, mesh_offset)
    return cache[path]


def inside_container_bounds(mesh_world, container_dims, eps=1e-6):
    mins, maxs = mesh_world.bounds
    if np.any(mins < -eps):
        return False
    if np.any(maxs - container_dims > eps):
        return False
    return True


def compute_stable_orientations(mesh, allow_stable_poses, max_poses=4):
    oriented = []
    try:
        if not allow_stable_poses:
            raise RuntimeError("stable poses disabled")
        from trimesh.poses import compute_stable_poses

        Ts, probs = compute_stable_poses(mesh, sigma=0.0, n_samples=1)
        order = np.argsort(-probs)[:max_poses]
        for i in order:
            m = mesh.copy()
            m.apply_transform(Ts[i])
            oriented.append((m, Ts[i]))
    except Exception:
        def _Rx90():
            M = np.eye(4)
            M[1, 1] = 0.0
            M[1, 2] = -1.0
            M[2, 1] = 1.0
            M[2, 2] = 0.0
            return M

        def _Ry90():
            M = np.eye(4)
            M[0, 0] = 0.0
            M[0, 2] = 1.0
            M[2, 0] = -1.0
            M[2, 2] = 0.0
            return M

        def _Rz90():
            M = np.eye(4)
            M[0, 0] = 0.0
            M[0, 1] = -1.0
            M[1, 0] = 1.0
            M[1, 1] = 0.0
            return M

        Rs = [np.eye(4), _Rx90(), _Ry90(), _Rz90()]
        for R in Rs[:max_poses]:
            m = mesh.copy()
            m.apply_transform(R)
            oriented.append((m, R))

    return oriented


def aabb_inside_container(bounds_local, translation, container_dims, eps=1e-6):
    mins = bounds_local[0] + translation
    maxs = bounds_local[1] + translation
    if np.any(mins < -eps):
        return False
    if np.any(maxs - container_dims > eps):
        return False
    return True


def get_oriented_variants(mesh_path, mesh, cache, allow_stable_poses, max_poses=4):
    if mesh_path in cache:
        return cache[mesh_path]

    variants = []
    for oriented_mesh, T_orient in compute_stable_orientations(mesh, allow_stable_poses, max_poses=max_poses):
        bounds = oriented_mesh.bounds.copy()
        variants.append({
            "mesh": oriented_mesh,
            "bounds": bounds,
            "offset": bounds[0].copy(),
            "size": oriented_mesh.extents.copy(),
            "T_orient": T_orient.copy(),
        })

    if not variants:
        oriented_mesh = mesh.copy()
        bounds = oriented_mesh.bounds.copy()
        variants.append({
            "mesh": oriented_mesh,
            "bounds": bounds,
            "offset": bounds[0].copy(),
            "size": oriented_mesh.extents.copy(),
            "T_orient": np.eye(4),
        })

    cache[mesh_path] = variants
    return variants
