#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# CRISP fallback 2.5D terrain mesh from nksr_input/pointcloud_world.npz\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _write_urdf(path: Path) -> None:
    robot = ET.Element("robot", {"name": "scene"})
    link = ET.SubElement(robot, "link", {"name": "scene_link"})
    for tag in ("visual", "collision"):
        elem = ET.SubElement(link, tag)
        geom = ET.SubElement(elem, "geometry")
        ET.SubElement(geom, "mesh", {"filename": "scene_mesh_sqs.obj"})
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def _robust_bounds(points: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(points[:, :2], low, axis=0)
    hi = np.percentile(points[:, :2], high, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def build_height_mesh(
    points: np.ndarray,
    resolution: float,
    z_quantile: float,
    min_points_per_cell: int,
    percentile_low: float,
    percentile_high: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        raise ValueError("pointcloud has no finite points")

    xy_min, xy_max = _robust_bounds(points, percentile_low, percentile_high)
    valid = (
        (points[:, 0] >= xy_min[0])
        & (points[:, 0] <= xy_max[0])
        & (points[:, 1] >= xy_min[1])
        & (points[:, 1] <= xy_max[1])
    )
    points = points[valid]
    if points.shape[0] == 0:
        raise ValueError("robust crop removed all points")

    nx = int(np.ceil((xy_max[0] - xy_min[0]) / resolution)) + 1
    ny = int(np.ceil((xy_max[1] - xy_min[1]) / resolution)) + 1
    if nx < 2 or ny < 2:
        raise ValueError(f"invalid grid size: nx={nx}, ny={ny}")

    ix = np.clip(((points[:, 0] - xy_min[0]) / resolution).astype(np.int32), 0, nx - 1)
    iy = np.clip(((points[:, 1] - xy_min[1]) / resolution).astype(np.int32), 0, ny - 1)
    flat = iy * nx + ix
    z = points[:, 2].astype(np.float32)

    order = np.lexsort((z, flat))
    flat_sorted = flat[order]
    z_sorted = z[order]
    unique, starts, counts = np.unique(flat_sorted, return_index=True, return_counts=True)

    z_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    support = np.zeros((ny, nx), dtype=bool)
    q = float(np.clip(z_quantile, 0.0, 1.0))
    for cell, start, count in zip(unique, starts, counts):
        if count < min_points_per_cell:
            continue
        pick = start + min(count - 1, int(round((count - 1) * q)))
        cy, cx = divmod(int(cell), nx)
        z_grid[cy, cx] = z_sorted[pick]
        support[cy, cx] = True

    if not support.any():
        raise ValueError("no grid cells had enough points for fallback mesh")

    _, nearest = ndimage.distance_transform_edt(~support, return_indices=True)
    z_grid = z_grid[nearest[0], nearest[1]].astype(np.float32)

    xs = xy_min[0] + np.arange(nx, dtype=np.float32) * resolution
    ys = xy_min[1] + np.arange(ny, dtype=np.float32) * resolution
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.stack([xx, yy, z_grid], axis=-1).reshape(-1, 3).astype(np.float32)

    faces = []
    for y in range(ny - 1):
        row = y * nx
        next_row = (y + 1) * nx
        for x in range(nx - 1):
            a = row + x
            b = row + x + 1
            c = next_row + x
            d = next_row + x + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    faces_arr = np.asarray(faces, dtype=np.int32)
    meta = {
        "resolution": resolution,
        "z_quantile": z_quantile,
        "min_points_per_cell": min_points_per_cell,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "input_points_after_crop": int(points.shape[0]),
        "grid_shape_ny_nx": [int(ny), int(nx)],
        "supported_cells": int(support.sum()),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces_arr.shape[0]),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }
    return vertices, faces_arr, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a CRISP-compatible fallback scene_mesh_sqs from pointcloud_world.npz."
    )
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--hmr-type", default="gv")
    parser.add_argument("--scene-root", type=Path, default=REPO_ROOT / "results" / "output" / "scene")
    parser.add_argument("--resolution", type=float, default=0.06)
    parser.add_argument("--z-quantile", type=float, default=0.08)
    parser.add_argument("--min-points-per-cell", type=int, default=25)
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seq_root = args.scene_root / args.seq_name / args.hmr_type
    pointcloud_path = seq_root / "nksr_input" / "pointcloud_world.npz"
    out_dir = seq_root / "scene_mesh_sqs"
    mesh_path = out_dir / "scene_mesh_sqs.obj"
    if mesh_path.exists() and not args.force:
        raise FileExistsError(f"fallback mesh already exists; pass --force: {mesh_path}")
    if not pointcloud_path.exists():
        raise FileNotFoundError(f"missing pointcloud: {pointcloud_path}")

    payload = np.load(pointcloud_path, allow_pickle=True)
    points = np.asarray(payload["points"], dtype=np.float32)
    vertices, faces, meta = build_height_mesh(
        points=points,
        resolution=args.resolution,
        z_quantile=args.z_quantile,
        min_points_per_cell=args.min_points_per_cell,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pieces").mkdir(exist_ok=True)
    _write_obj(mesh_path, vertices, faces)
    _write_urdf(out_dir / "scene_mesh_sqs.urdf")
    empty_params = np.zeros((0, 11), dtype=np.float32)
    np.save(out_dir / "sqs_params.npy", empty_params)
    np.savez_compressed(out_dir / "sqs_params.npz", params=empty_params)
    meta.update(
        {
            "sequence_name": args.seq_name,
            "hmr_type": args.hmr_type,
            "source_pointcloud": str(pointcloud_path),
            "mesh_path": str(mesh_path),
            "urdf_path": str(out_dir / "scene_mesh_sqs.urdf"),
            "note": "fallback 2.5D mesh; not CRISP SQS planes",
        }
    )
    (out_dir / "fallback_scene_mesh_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
