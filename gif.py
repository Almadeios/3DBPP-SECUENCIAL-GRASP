import argparse
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyrender
import trimesh

# Mantiene colores fijos para cada objeto durante todo el GIF
OBJECT_COLORS = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--output", default="visualizacion")
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def fixed_color(index):

    if index not in OBJECT_COLORS:

        OBJECT_COLORS[index] = [
            random.randint(150, 255),
            random.randint(150, 255),
            random.randint(150, 255),
            255
        ]

    return OBJECT_COLORS[index]


def look_at(eye, target):

    forward = target - eye
    forward /= np.linalg.norm(forward)

    up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    pose = np.eye(4)

    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye

    return pose


def build_scene(placed_subset, obj_dir, container_dims):

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[0.4, 0.4, 0.4]
    )

    container_box = trimesh.creation.box(
        extents=container_dims
    )

    container_box.visual.face_colors = [180, 220, 255, 15]

    container_tf = np.eye(4)
    container_tf[:3, 3] = container_dims / 2.0

    scene.add(
        pyrender.Mesh.from_trimesh(
            container_box,
            smooth=False
        ),
        pose=container_tf
    )

    # ---------- ARISTAS VISIBLES ----------
    corners = np.array([
        [0, 0, 0],
        [container_dims[0], 0, 0],
        [container_dims[0], container_dims[1], 0],
        [0, container_dims[1], 0],

        [0, 0, container_dims[2]],
        [container_dims[0], 0, container_dims[2]],
        [container_dims[0], container_dims[1], container_dims[2]],
        [0, container_dims[1], container_dims[2]]
    ])

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for a, b in edges:

        edge = trimesh.creation.cylinder(
            radius=0.0015,
            segment=[
                corners[a],
                corners[b]
            ]
        )

        edge.visual.face_colors = [0, 0, 0, 255]

        scene.add(
            pyrender.Mesh.from_trimesh(
                edge,
                smooth=False
            )
        )

    # Wireframe del contenedor
    container_wire = pyrender.Mesh.from_trimesh(
        container_box,
        smooth=False,
        wireframe=True
    )

    scene.add(
        container_wire,
        pose=container_tf
    )

    for idx, item in enumerate(placed_subset):

        obj_path = obj_dir / Path(item["id"])

        if not obj_path.exists():
            print("No encontrado:", obj_path)
            continue

        mesh = trimesh.load(
            str(obj_path),
            force="mesh"
        )

        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()

        color = fixed_color(idx)

        mesh.visual.face_colors = np.tile(
            color,
            (len(mesh.faces), 1)
        )

        tf = np.array(
            item["transform_matrix"]
        )

        scene.add(
            pyrender.Mesh.from_trimesh(
                mesh,
                smooth=False
            ),
            pose=tf
        )

    return scene


def compute_camera(container_dims):

    center = container_dims / 2.0

    extent = np.max(container_dims)

    eye = center + np.array([
        extent * 1.2,
        -extent * 1.2,
        extent * 0.9
    ])

    return look_at(
        eye,
        center
    )


def render_scene(scene, camera_pose):

    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 3.0
    )

    scene.add(
        camera,
        pose=camera_pose
    )

    light = pyrender.DirectionalLight(
        color=np.ones(3),
        intensity=5.0
    )

    scene.add(
        light,
        pose=camera_pose
    )

    renderer = pyrender.OffscreenRenderer(
        viewport_width=1280,
        viewport_height=720
    )

    color, _ = renderer.render(scene)

    renderer.delete()

    return color


def main():

    args = parse_args()

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = Path(args.json)

    with open(json_path, "r") as f:
        placed = json.load(f)

    dataset_name = json_path.parent.name

    obj_dir = (
        Path("dataset")
        / dataset_name
        / "shape_vhacd"
    )

    container_dims = np.array(
        [320, 320, 300]
    ) / 1000.0

    camera_pose = compute_camera(
        container_dims
    )

    frames = []

    total = len(placed)

    for i in range(total):

        current = placed[: i + 1]

        scene = build_scene(
            current,
            obj_dir,
            container_dims
        )

        image = render_scene(
            scene,
            camera_pose
        )

        frame_file = (
            output_dir
            / f"frame_{i:03d}.png"
        )

        imageio.imwrite(
            frame_file,
            image
        )

        frames.append(image)

        print(f"[{i+1}/{total}]")

    gif_file = (
        output_dir
        / "packing.gif"
    )

    imageio.mimsave(
        gif_file,
        frames,
        fps=args.fps,
        loop=0
    )

    imageio.imwrite(
        output_dir / "vista_final.png",
        frames[-1]
    )

    print()
    print("GIF generado:")
    print(gif_file)


if __name__ == "__main__":
    main()
