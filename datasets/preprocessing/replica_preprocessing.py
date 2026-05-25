"""Convert a Replica scene's mesh.ply into the (N, 9) .npy that Mask3D's
test pipeline expects: [x, y, z, r, g, b, nx, ny, nz] as float32.

Mirrors scannet_preprocessing.process_file for test-mode (no GT labels):
    coords, feats, _ = load_ply_with_normals(mesh.ply)
    points = np.hstack((coords, feats))  # (N, 9)
    np.save(out, points.astype(np.float32))

Usage:
    python -m datasets.preprocessing.replica_preprocessing \\
        --scene_dir /home/olaf/tum_dilab_ss26/datasets/office_0 \\
        --out_dir   ./data/processed/replica
"""

from pathlib import Path

import numpy as np
from fire import Fire

from utils.point_cloud_utils import load_ply_with_normals


def convert_scene(scene_dir: str, out_dir: str) -> str:
    scene_dir = Path(scene_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = scene_dir / "mesh.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh.ply not found under {scene_dir}")

    coords, feats, _ = load_ply_with_normals(mesh_path)
    points = np.hstack((coords, feats)).astype(np.float32)

    out_path = out_dir / f"{scene_dir.name}.npy"
    np.save(out_path, points)
    return str(out_path)


def main(
    scene_dir: str,
    out_dir: str = "./data/processed/replica",
):
    out_path = convert_scene(scene_dir, out_dir)
    arr = np.load(out_path)
    print(f"wrote {out_path}")
    print(f"  shape={arr.shape} dtype={arr.dtype}")
    print(f"  xyz  range: min={arr[:, :3].min(0)} max={arr[:, :3].max(0)}")
    print(f"  rgb  range: min={arr[:, 3:6].min(0)} max={arr[:, 3:6].max(0)}")
    print(f"  norm range: min={arr[:, 6:9].min(0)} max={arr[:, 6:9].max(0)}")


if __name__ == "__main__":
    Fire(main)
