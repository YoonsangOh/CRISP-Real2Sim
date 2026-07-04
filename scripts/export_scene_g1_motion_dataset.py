#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Invalid mesh: {mesh_path}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a portable scene mesh + G1 motion dataset from a PyRoki retarget result."
    )
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument(
        "--retarget-mode",
        default="simple",
        help="PyRoki retarget result to export. Defaults to the stable no-heightmap-contact mode.",
    )
    parser.add_argument(
        "--retarget-root",
        type=Path,
        default=REPO_ROOT / "results" / "output" / "g1_pyroki",
    )
    parser.add_argument(
        "--post-scene-root",
        type=Path,
        default=REPO_ROOT / "results" / "output" / "post_scene",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "output" / "scene_g1_motion_dataset",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    retarget_file = (
        args.retarget_root
        / args.seq_name
        / args.hmr_type
        / f"{args.seq_name}_scene_g1_pyroki_{args.retarget_mode}.npz"
    ).resolve()
    post_scene_dir = (args.post_scene_root / args.seq_name / args.hmr_type).resolve()
    mesh_path = post_scene_dir / "scene_mesh_sqs" / "scene_mesh_sqs.obj"
    urdf_path = post_scene_dir / "scene_mesh_sqs" / "scene_mesh_sqs.urdf"

    if not retarget_file.exists():
        raise FileNotFoundError(f"Missing retarget file: {retarget_file}")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing scene mesh: {mesh_path}")

    data = np.load(retarget_file, allow_pickle=True)
    mesh = load_mesh(mesh_path)

    output_dir = args.output_root / args.seq_name / args.hmr_type
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.seq_name}_scene_g1_motion.npz"
    if output_file.exists() and not args.force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_file}")

    np.savez_compressed(
        output_file,
        schema_version=np.array("scene_g1_motion_v1"),
        sequence_name=np.array(args.seq_name),
        hmr_type=np.array(args.hmr_type),
        coordinate_frame=np.array("world_z_up_scene_grounded_gravity_down_negative_z"),
        gravity_vector_world=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        fps=np.asarray(data["fps"], dtype=np.float32),
        dt=np.float32(1.0 / float(data["fps"])),
        frame_count=np.asarray(data["frame_count"], dtype=np.int32),
        robot_name=np.array("unitree_g1_29dof"),
        g1_urdf_description=np.asarray(data["g1_urdf_description"]),
        g1_dof_names=np.asarray(data["g1_dof_names"]),
        g1_link_names=np.asarray(data["g1_link_names"]),
        g1_root_pos=np.asarray(data["g1_root_pos"], dtype=np.float32),
        g1_root_quat_wxyz=np.asarray(data["g1_root_quat_wxyz"], dtype=np.float32),
        g1_dof_pos=np.asarray(data["g1_dof_pos"], dtype=np.float32),
        g1_dof_vel=np.asarray(data["g1_dof_vel"], dtype=np.float32),
        g1_root_lin_vel=np.asarray(data["g1_root_lin_vel"], dtype=np.float32),
        g1_qpos=np.asarray(data["g1_qpos"], dtype=np.float32),
        scene_mesh_vertices=np.asarray(mesh.vertices, dtype=np.float32),
        scene_mesh_faces=np.asarray(mesh.faces, dtype=np.int32),
        scene_mesh_bounds=np.asarray(mesh.bounds, dtype=np.float32),
        scene_mesh_source=np.array(str(mesh_path)),
        scene_urdf_source=np.array(str(urdf_path) if urdf_path.exists() else ""),
        world_rotation=np.asarray(data["world_rotation"], dtype=np.float32),
        shared_translation=np.asarray(data["shared_translation"], dtype=np.float32),
        source_retarget_file=np.array(str(retarget_file)),
        source_post_scene=np.array(str(post_scene_dir)),
    )

    print(f"saved={output_file}")
    print(f"scene_vertices={mesh.vertices.shape[0]} scene_faces={mesh.faces.shape[0]}")
    print(f"frames={data['frame_count'].item()} dof={len(data['g1_dof_names'])} fps={float(data['fps'])}")


if __name__ == "__main__":
    main()
