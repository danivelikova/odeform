"""Align HouseCat6D shoe meshes to the flip-flop of the Contact-Force dataset (Fig. 5).

The HouseCat6D object meshes (https://sites.google.com/view/housecat6d) are rotated
into the Blender frame of the Contact-Force dataset (+90 deg about x, then +90 deg about
z) and scaled/translated so that their axis-aligned bounding box matches the one of the
reference flip-flop mesh. The aligned meshes can be deformed with ``infer.py``.

Example:
    python scripts/align_housecat6d.py \
        --meshes /path/to/housecat6d/obj_models/shoe \
        --reference data/contact_force/flipflop/1112_224643/1112_224643_1.ply \
        --output data/housecat6d
"""
import argparse
import glob
import os

import numpy as np
import open3d as o3d


def rotation(axis, degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def align(vertices, reference):
    v = vertices @ rotation("x", 90).T
    v = v @ rotation("z", 90).T
    scale = np.ptp(reference, axis=0) / np.ptp(v, axis=0)
    return (v - v.mean(axis=0)) * scale + reference.mean(axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meshes", required=True, help="folder with HouseCat6D shoe .obj files")
    p.add_argument("--reference", required=True, help="rest mesh of a Contact-Force flip-flop sequence")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    reference = np.asarray(o3d.io.read_triangle_mesh(args.reference).vertices)
    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.meshes, "*.obj")))
    if not files:
        raise SystemExit(f"No .obj files found in {args.meshes}")
    for path in files:
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.vertices = o3d.utility.Vector3dVector(align(np.asarray(mesh.vertices), reference))
        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.output, f"aligned_mesh_{name}.ply")
        o3d.io.write_triangle_mesh(out, mesh, write_ascii=False)
        print(f"{path} -> {out}")


if __name__ == "__main__":
    main()
