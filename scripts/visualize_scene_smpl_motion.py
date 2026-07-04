#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POST_SCENE = REPO_ROOT / "results/output/post_scene/climbing_down_stairs_001/gv"


SMPL22_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (9, 12),
    (12, 15),
    (9, 13),
    (9, 14),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (7, 10),
    (8, 11),
]


def load_scene_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise ValueError(f"Invalid scene mesh: {mesh_path}")
    return mesh


def set_axes_equal(ax: plt.Axes, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    radius = float(np.max(bounds[1] - bounds[0]) * 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius * 0.35), center[2] + radius * 0.85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CRISP post_scene scene + SMPL human joints to MP4.")
    parser.add_argument("--post-scene", type=Path, default=DEFAULT_POST_SCENE)
    parser.add_argument("--seq-name", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--max-scene-faces", type=int, default=2500)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    args = parser.parse_args()

    post_scene = args.post_scene.resolve()
    seq_name = args.seq_name or post_scene.parent.name
    joints_file = post_scene / "hmr" / f"{seq_name}.npz"
    motion_file = post_scene / "hmr" / "human_motion.npz"
    mesh_file = post_scene / "scene_mesh_sqs" / "scene_mesh_sqs.obj"
    if not joints_file.exists():
        raise FileNotFoundError(f"Missing SMPL joints file: {joints_file}")
    if not motion_file.exists():
        raise FileNotFoundError(f"Missing human motion file: {motion_file}")

    joints = np.load(joints_file, allow_pickle=True)["global_joint_positions"].astype(np.float32)
    motion = np.load(motion_file, allow_pickle=True)
    fps = int(motion["mocap_framerate"]) if "mocap_framerate" in motion else args.fps
    scene = load_scene_mesh(mesh_file)
    vertices = np.asarray(scene.vertices, dtype=np.float32)
    faces = np.asarray(scene.faces, dtype=np.int32)
    if len(faces) > args.max_scene_faces:
        face_indices = np.linspace(0, len(faces) - 1, args.max_scene_faces).astype(np.int32)
        faces_to_draw = faces[face_indices]
    else:
        faces_to_draw = faces

    output = args.output
    if output is None:
        output = post_scene / "scene_smpl_motion.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    all_bounds = np.vstack([scene.bounds, joints.reshape(-1, 3).min(axis=0), joints.reshape(-1, 3).max(axis=0)])
    bounds = np.vstack([all_bounds.min(axis=0), all_bounds.max(axis=0)])
    frame_indices = list(range(0, joints.shape[0], max(1, args.stride)))

    writer = imageio.get_writer(output, fps=args.fps or fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for render_idx, frame_idx in enumerate(frame_indices):
            fig = plt.figure(figsize=(8, 7), dpi=args.dpi)
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=args.elev, azim=args.azim)
            set_axes_equal(ax, bounds)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"{seq_name} | SMPL human motion | frame {frame_idx}/{joints.shape[0] - 1}")

            mesh = Poly3DCollection(vertices[faces_to_draw], alpha=0.18, linewidths=0.05)
            mesh.set_facecolor((0.45, 0.50, 0.48, 0.18))
            mesh.set_edgecolor((0.25, 0.28, 0.26, 0.12))
            ax.add_collection3d(mesh)

            pts = joints[frame_idx]
            for a, b in SMPL22_EDGES:
                seg = pts[[a, b]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#2ca02c", linewidth=2.8)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18, color="#ff7f0e", depthshade=True)

            ax.text2D(0.02, 0.96, f"source=CRISP post_scene, fps={fps}, z-up", transform=ax.transAxes, fontsize=9)
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
            plt.close(fig)
            if render_idx == 0 or (render_idx + 1) % 25 == 0 or render_idx + 1 == len(frame_indices):
                print(f"rendered {render_idx + 1}/{len(frame_indices)} frames")
    finally:
        writer.close()

    print(f"saved={output}")


if __name__ == "__main__":
    main()
