#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import trimesh
from robot_descriptions.loaders.yourdfpy import load_robot_description


REPO_ROOT = Path(__file__).resolve().parents[1]
PYROKI_ROOT = REPO_ROOT / "third_party" / "pyroki"
PYROKI_EXAMPLES = PYROKI_ROOT / "examples"
SMPL_45 = 45


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise ValueError(f"Empty or unsupported scene mesh: {path}")
    return mesh


def _validate_alignment(post_root: Path, mesh: trimesh.Trimesh, tolerance: float) -> dict[str, Any]:
    motion = np.load(post_root / "hmr" / "human_motion.npz", allow_pickle=True)
    rotation = np.asarray(motion["world_rotation"], dtype=np.float32)
    shared_translation = np.asarray(motion["shared_translation"], dtype=np.float32)
    verts = mesh.vertices.view(np.ndarray)
    z_min = float(verts[:, 2].min())
    z_max = float(verts[:, 2].max())
    det = float(np.linalg.det(rotation))
    orth_err = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    ok = abs(z_min) <= tolerance and abs(det - 1.0) <= 1e-3 and orth_err <= 1e-3
    return {
        "ok": ok,
        "z_min": z_min,
        "z_max": z_max,
        "rotation_det": det,
        "rotation_orth_err": orth_err,
        "world_rotation": rotation,
        "shared_translation": shared_translation,
    }


def _load_smpl_keypoints_45(post_root: Path, seq_name: str) -> tuple[np.ndarray, np.ndarray]:
    joints_path = post_root / "hmr" / f"{seq_name}.npz"
    if not joints_path.exists():
        raise FileNotFoundError(f"Missing post-scene joint file: {joints_path}")
    payload = np.load(joints_path, allow_pickle=True)
    joints22 = np.asarray(payload["global_joint_positions"], dtype=np.float32)
    if joints22.ndim != 3 or joints22.shape[1:] != (22, 3):
        raise ValueError(f"Expected global_joint_positions shape (T,22,3), got {joints22.shape}")

    keypoints45 = np.zeros((joints22.shape[0], SMPL_45, 3), dtype=np.float32)
    keypoints45[:, :22, :] = joints22
    return keypoints45, joints22


def _make_contacts(
    heightmap: pk.collision.Heightmap,
    keypoints45: np.ndarray,
    threshold: float,
    velocity_threshold: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    helper = _load_module(PYROKI_EXAMPLES / "retarget_helpers" / "_utils.py", "pyroki_retarget_utils")
    left_idx = helper.SMPL_JOINT_NAMES.index("left_foot")
    right_idx = helper.SMPL_JOINT_NAMES.index("right_foot")
    left = jnp.asarray(keypoints45[:, left_idx, :])
    right = jnp.asarray(keypoints45[:, right_idx, :])
    left_projected = np.asarray(heightmap.project_points(left))
    right_projected = np.asarray(heightmap.project_points(right))
    left_dist = keypoints45[:, left_idx, 2] - left_projected[:, 2]
    right_dist = keypoints45[:, right_idx, 2] - right_projected[:, 2]
    dt = 1.0 / float(fps)
    left_speed = np.linalg.norm(np.gradient(keypoints45[:, left_idx, :], dt, axis=0), axis=1)
    right_speed = np.linalg.norm(np.gradient(keypoints45[:, right_idx, :], dt, axis=0), axis=1)
    return (
        (left_dist <= threshold) & (left_speed <= velocity_threshold),
        (right_dist <= threshold) & (right_speed <= velocity_threshold),
    )


def _solve(
    mode: str,
    robot: pk.Robot,
    target_keypoints: np.ndarray,
    heightmap: pk.collision.Heightmap,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    helper = _load_module(PYROKI_EXAMPLES / "retarget_helpers" / "_utils.py", "pyroki_retarget_utils")
    smpl_indices, g1_indices = helper.get_humanoid_retarget_indices()
    smpl_mask = helper.create_conn_tree(robot, g1_indices)

    if mode == "simple":
        module = _load_module(PYROKI_EXAMPLES / "10_humanoid_retargeting.py", "pyroki_humanoid_simple")
        weights = module.RetargetingWeights(local_alignment=2.0, global_alignment=1.0)
        Ts_world_root, joints = module.solve_retargeting(
            robot=robot,
            target_keypoints=jnp.asarray(target_keypoints),
            smpl_joint_retarget_indices=smpl_indices,
            g1_joint_retarget_indices=g1_indices,
            smpl_mask=smpl_mask,
            weights=weights,
        )
    else:
        module = _load_module(PYROKI_EXAMPLES / "12_humanoid_retargeting_fancy.py", "pyroki_humanoid_fancy")
        robot_coll = pk.collision.RobotCollision.from_urdf(load_robot_description("g1_description"))
        left_idx = helper.SMPL_JOINT_NAMES.index("left_foot")
        right_idx = helper.SMPL_JOINT_NAMES.index("right_foot")
        left_foot = heightmap.project_points(jnp.asarray(target_keypoints[:, left_idx, :]))
        right_foot = heightmap.project_points(jnp.asarray(target_keypoints[:, right_idx, :]))
        weights = module.RetargetingWeights(
            local_alignment=2.0,
            global_alignment=1.0,
            floor_contact=1.0,
            root_smoothness=1.0,
            foot_skating=1.0,
            world_collision=1.0,
        )
        Ts_world_root, joints = module.solve_retargeting(
            robot=robot,
            robot_coll=robot_coll,
            target_keypoints=jnp.asarray(target_keypoints),
            is_left_foot_contact=jnp.asarray(left_contact),
            is_right_foot_contact=jnp.asarray(right_contact),
            left_foot_keypoints=left_foot,
            right_foot_keypoints=right_foot,
            smpl_joint_retarget_indices=smpl_indices,
            g1_joint_retarget_indices=g1_indices,
            smpl_mask=smpl_mask,
            heightmap=heightmap,
            weights=weights,
        )
    return np.asarray(Ts_world_root.wxyz_xyz), np.asarray(joints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget a CRISP post_scene sequence to Unitree G1 29-DoF with PyRoki.")
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument(
        "--mode",
        choices=("simple", "fancy"),
        default="simple",
        help=(
            "Use simple by default. The fancy mode adds scene-heightmap contact/collision "
            "costs and is unsafe for CRISP SQS meshes that include stairs/walls/background planes."
        ),
    )
    parser.add_argument("--post-scene-root", type=Path, default=REPO_ROOT / "results" / "output" / "post_scene")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results" / "output" / "g1_pyroki")
    parser.add_argument("--heightmap-resolution", type=float, default=0.05)
    parser.add_argument("--contact-threshold", type=float, default=0.08)
    parser.add_argument("--contact-velocity-threshold", type=float, default=0.35)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    post_root = (args.post_scene_root / args.seq_name / args.hmr_type).resolve()
    scene_root = post_root / "scene_mesh_sqs"
    mesh_path = scene_root / "scene_mesh_sqs.obj"
    if not post_root.exists():
        raise FileNotFoundError(f"Missing CRISP post_scene: {post_root}")
    mesh = _load_mesh(mesh_path)
    alignment = _validate_alignment(post_root, mesh, args.alignment_tolerance)
    if not alignment["ok"]:
        raise RuntimeError(f"Post-scene alignment check failed: {json.dumps({k: v for k, v in alignment.items() if k not in ('world_rotation','shared_translation')})}")

    keypoints45, joints22 = _load_smpl_keypoints_45(post_root, args.seq_name)
    fps = float(np.load(post_root / "hmr" / "human_motion.npz", allow_pickle=True)["mocap_framerate"])
    heightmap = pk.collision.Heightmap.from_trimesh(mesh, resolution=args.heightmap_resolution)
    left_contact, right_contact = _make_contacts(
        heightmap,
        keypoints45,
        args.contact_threshold,
        args.contact_velocity_threshold,
        fps,
    )

    urdf = load_robot_description("g1_description")
    robot = pk.Robot.from_urdf(urdf)
    if len(robot.joints.actuated_names) != 29:
        raise RuntimeError(f"Expected G1 29-DoF, got {len(robot.joints.actuated_names)} actuated joints")

    out_dir = args.output_root / args.seq_name / args.hmr_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.seq_name}_scene_g1_pyroki_{args.mode}.npz"
    if out_path.exists() and not args.force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {out_path}")

    root_wxyz_xyz, dof_pos = _solve(args.mode, robot, keypoints45, heightmap, left_contact, right_contact)
    dt = 1.0 / fps
    dof_vel = np.gradient(dof_pos, dt, axis=0).astype(np.float32)
    root_pos = root_wxyz_xyz[:, 4:].astype(np.float32)
    root_quat_wxyz = root_wxyz_xyz[:, :4].astype(np.float32)
    root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)

    np.savez_compressed(
        out_path,
        schema_version=np.array("crisp_scene_g1_pyroki_v1"),
        sequence_name=np.array(args.seq_name),
        hmr_type=np.array(args.hmr_type),
        coordinate_frame=np.array("world_z_up_scene_grounded_gravity_down_negative_z"),
        gravity_vector_world=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        fps=np.float32(fps),
        frame_count=np.int32(dof_pos.shape[0]),
        retarget_backend=np.array("pyroki"),
        retarget_mode=np.array(args.mode),
        pyroki_git_commit=np.array((PYROKI_ROOT / ".git").exists() and __import__("subprocess").check_output(["git", "-C", str(PYROKI_ROOT), "rev-parse", "HEAD"], text=True).strip() or ""),
        g1_urdf_description=np.array("g1_description"),
        g1_dof_names=np.asarray(robot.joints.actuated_names),
        g1_link_names=np.asarray(robot.links.names),
        g1_root_pos=root_pos,
        g1_root_quat_wxyz=root_quat_wxyz,
        g1_dof_pos=dof_pos.astype(np.float32),
        g1_dof_vel=dof_vel,
        g1_root_lin_vel=root_lin_vel,
        g1_qpos=np.concatenate([root_pos, root_quat_wxyz, dof_pos.astype(np.float32)], axis=1),
        human_keypoints_smpl45=keypoints45,
        human_joints_smpl22=joints22,
        left_foot_contact=left_contact,
        right_foot_contact=right_contact,
        scene_mesh_obj=np.array(str(mesh_path)),
        scene_urdf=np.array(str(scene_root / "scene_mesh_sqs.urdf")),
        scene_sqs_params=np.array(str(scene_root / "sqs_params.npz")),
        scene_bounds=mesh.bounds.astype(np.float32),
        scene_heightmap=np.asarray(heightmap.height_data, dtype=np.float32),
        scene_heightmap_size=np.asarray(heightmap.size, dtype=np.float32),
        scene_heightmap_pose_wxyz_xyz=np.asarray(heightmap.pose.wxyz_xyz, dtype=np.float32),
        world_rotation=np.asarray(alignment["world_rotation"], dtype=np.float32),
        shared_translation=np.asarray(alignment["shared_translation"], dtype=np.float32),
        source_post_scene=np.array(str(post_root)),
    )
    print(f"saved={out_path}")
    print(f"frames={dof_pos.shape[0]} dof={dof_pos.shape[1]} mode={args.mode}")
    print(f"alignment_z_min={alignment['z_min']:.8f} rotation_det={alignment['rotation_det']:.8f}")
    print(f"heightmap_shape={np.asarray(heightmap.height_data).shape} device={jax.devices()[0]}")


if __name__ == "__main__":
    main()
