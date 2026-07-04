#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import jaxlie
import matplotlib
import numpy as np
import pyroki as pk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from robot_descriptions.loaders.yourdfpy import load_robot_description

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPO_ROOT
    / "results/output/scene_g1_motion_dataset/climbing_down_stairs_001/gv/climbing_down_stairs_001_scene_g1_motion.npz"
)


SKELETON_EDGES = [
    ("pelvis", "pelvis_contour_link"),
    ("pelvis_contour_link", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis_contour_link", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("pelvis_contour_link", "waist_yaw_link"),
    ("waist_yaw_link", "waist_roll_link"),
    ("waist_roll_link", "torso_link"),
    ("torso_link", "head_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("left_wrist_roll_link", "left_wrist_pitch_link"),
    ("left_wrist_pitch_link", "left_wrist_yaw_link"),
    ("left_wrist_yaw_link", "left_rubber_hand"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
    ("right_wrist_roll_link", "right_wrist_pitch_link"),
    ("right_wrist_pitch_link", "right_wrist_yaw_link"),
    ("right_wrist_yaw_link", "right_rubber_hand"),
]


def compute_link_positions(data: np.lib.npyio.NpzFile) -> np.ndarray:
    robot = pk.Robot.from_urdf(load_robot_description("g1_description"))
    dof_pos = jnp.asarray(data["g1_dof_pos"])
    root_wxyz_xyz = jnp.asarray(np.concatenate([data["g1_root_quat_wxyz"], data["g1_root_pos"]], axis=1))

    t_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=dof_pos))
    t_world_root = jaxlie.SE3(root_wxyz_xyz)
    t_world_root = jax.tree.map(lambda x: x[:, None, ...], t_world_root)
    return np.asarray((t_world_root @ t_root_link).translation(), dtype=np.float32)


def set_axes_equal(ax: plt.Axes, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    radius = float(np.max(bounds[1] - bounds[0]) * 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius * 0.35), center[2] + radius * 0.85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a scene+G1 motion dataset to MP4.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame.")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--max-scene-faces", type=int, default=2500)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    output = args.output
    if output is None:
        output = args.dataset.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    link_positions = compute_link_positions(data)
    link_names = [str(x) for x in data["g1_link_names"]]
    link_to_idx = {name: idx for idx, name in enumerate(link_names)}
    edges = [(link_to_idx[a], link_to_idx[b]) for a, b in SKELETON_EDGES if a in link_to_idx and b in link_to_idx]

    vertices = np.asarray(data["scene_mesh_vertices"], dtype=np.float32)
    faces = np.asarray(data["scene_mesh_faces"], dtype=np.int32)
    if len(faces) > args.max_scene_faces:
        face_indices = np.linspace(0, len(faces) - 1, args.max_scene_faces).astype(np.int32)
        faces_to_draw = faces[face_indices]
    else:
        faces_to_draw = faces

    all_bounds = np.vstack([data["scene_mesh_bounds"], link_positions.reshape(-1, 3).min(axis=0), link_positions.reshape(-1, 3).max(axis=0)])
    bounds = np.vstack([all_bounds.min(axis=0), all_bounds.max(axis=0)])
    frame_indices = list(range(0, link_positions.shape[0], max(1, args.stride)))

    writer = imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for render_idx, frame_idx in enumerate(frame_indices):
            fig = plt.figure(figsize=(8, 7), dpi=args.dpi)
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=args.elev, azim=args.azim)
            set_axes_equal(ax, bounds)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"{data['sequence_name'].item()} | frame {frame_idx}/{link_positions.shape[0] - 1}")

            mesh = Poly3DCollection(vertices[faces_to_draw], alpha=0.18, linewidths=0.05)
            mesh.set_facecolor((0.45, 0.50, 0.48, 0.18))
            mesh.set_edgecolor((0.25, 0.28, 0.26, 0.12))
            ax.add_collection3d(mesh)

            pts = link_positions[frame_idx]
            for a, b in edges:
                seg = pts[[a, b]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#1f77b4", linewidth=2.4)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, color="#d62728", depthshade=True)

            ax.text2D(
                0.02,
                0.96,
                f"fps={float(data['fps']):.0f}, dof=29, z-up",
                transform=ax.transAxes,
                fontsize=9,
            )
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
