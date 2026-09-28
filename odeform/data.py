"""Datasets for the Contact-Force and Mass-Elastic experiments.

Expected layout (one folder per simulated sequence)::

    <dataset_root>/
        train_folders.txt  val_folders.txt  test_folders.txt
        <seq_id>/
            params.json         physical parameters of the sequence
            metadata.json       per-frame global offset of the object
            <seq_id>_<k>.ply    meshes (connectivity is used to export predictions)
            xyz_meshes/<seq_id>_<k>.xyz   per-frame vertex positions (+ normals)

All point clouds of a split are normalized with the same reference: the first frame of
the first sequence of that split (same procedure as for the paper experiments).
"""
import json
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

NORM_BUFFER_FACTOR = 2


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", s)]


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_points(path):
    """Loads vertex positions (N, 3) from a .xyz, .ply or .obj file."""
    if path.endswith(".xyz"):
        return np.genfromtxt(path)[:, :3].astype(np.float32)
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float32)
    return np.asarray(mesh.vertices, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
def normalization_params(ref_coords):
    """(coord_min, coord_max, ref_mean) such that the reference fits in [-1, 1]."""
    ref_mean = ref_coords.mean(axis=0, keepdims=True)
    centered = ref_coords - ref_mean
    return np.float32(centered.min()), np.float32(centered.max()), ref_mean.astype(np.float32)


def normalize(coords, coord_min, coord_max, ref_mean):
    coords_n = (coords - ref_mean - coord_min) / (coord_max - coord_min)
    return (coords_n - 0.5) * NORM_BUFFER_FACTOR


def denormalize(coords_n, coord_min, coord_max, ref_mean):
    coords = (coords_n / NORM_BUFFER_FACTOR + 0.5) * (coord_max - coord_min) + coord_min
    return coords + ref_mean


def normalize_mass_elastic_params(params):
    """mass and bending stiffness are sampled in [0, 3] and mapped to [-1, 1]."""
    return ((np.asarray(params, dtype=np.float32) / 3.0) - 0.5) * NORM_BUFFER_FACTOR


def denormalize_mass_elastic_params(params_n):
    return (np.asarray(params_n) / NORM_BUFFER_FACTOR + 0.5) * 3.0


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def get_splits(dataset_root, train_ratio=0.7, val_ratio=0.1, seed=42):
    """Returns (train, val, test) sequence folders.

    The split files shipped with the data are used when present. Otherwise the split is
    regenerated with the fixed seed used for the paper (identical result).
    """
    files = [os.path.join(dataset_root, f"{s}_folders.txt") for s in ("train", "val", "test")]
    if all(os.path.exists(f) for f in files):
        return tuple([l for l in open(f).read().splitlines() if l] for f in files)

    random.seed(seed)
    folders = sorted(f for f in os.listdir(dataset_root)
                     if os.path.isdir(os.path.join(dataset_root, f, "xyz_meshes")))
    random.shuffle(folders)
    n_train, n_val = int(train_ratio * len(folders)), int(val_ratio * len(folders))
    return (folders[:n_train], folders[n_train:n_train + n_val], folders[n_train + n_val:])


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
class DeformationSequenceDataset(Dataset):
    """Base class. Each item is one sequence restricted to a window of frames.

    Args:
        root:           dataset root folder.
        folders:        sequence folders to load (e.g. from ``get_splits``).
        seq_start_idx:  first frame of the window.
        sequence_length: number of frames in the window.
        fps:            frame rate used to compute the time stamps t = k / fps.
        frame_interval: keep only every ``frame_interval``-th frame of the window
                        (sparse supervision). 1 = all frames.
    """
    offset_dim = None
    offset_scale = None

    def __init__(self, root, folders, seq_start_idx=0, sequence_length=10, fps=10,
                 frame_interval=1, sort_folders=True, verbose=True):
        super().__init__()
        self.root = root
        self.folders = sorted(folders) if sort_folders else list(folders)
        frames = list(range(seq_start_idx, seq_start_idx + sequence_length))
        self.frame_ids = frames[::frame_interval]
        self.fps = fps
        self.sequences = []
        self.norm = None

        for folder in self.folders:
            xyz_dir = os.path.join(root, folder, "xyz_meshes")
            if not os.path.isdir(xyz_dir):
                continue
            entries = sorted(os.listdir(xyz_dir), key=natural_sort_key)
            if self.norm is None:
                # Reference = first frame of the first sequence of the split.
                self.norm = normalization_params(load_points(os.path.join(xyz_dir, entries[0])))
            parameters, offsets = self.read_sequence_info(os.path.join(root, folder))

            coords, times, offs, paths = [], [], [], []
            for i in self.frame_ids:
                path = os.path.join(xyz_dir, entries[i])
                coords.append(normalize(load_points(path), *self.norm))
                times.append(np.round(np.float32(i * (1 / fps)), 2))
                offs.append(self.offset_scale * np.atleast_1d(np.float32(offsets[i])))
                paths.append(path)
            self.sequences.append(dict(
                name=folder,
                coords=np.stack(coords).astype(np.float32),
                time=np.array(times, dtype=np.float32),
                offset=np.stack(offs).astype(np.float32),
                parameters=parameters,
                files=paths,
                mesh=self.rest_mesh_path(os.path.join(root, folder)),
            ))
        if verbose:
            print(f"Loaded {len(self.sequences)} sequences from {root} "
                  f"(frames {self.frame_ids[0]}..{self.frame_ids[-1]}, {len(self.frame_ids)} per sequence)")

    def read_sequence_info(self, seq_dir):
        raise NotImplementedError

    def rest_mesh_path(self, seq_dir):
        return None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in s.items()}


class ContactForceDataset(DeformationSequenceDataset):
    """Rigid projectile hitting a soft object (paper Sec. IV-A, Contact-Force).

    theta = (contact location xyz, force/contact vector xyz); offset = rigid xyz motion.
    """
    offset_dim = 3
    offset_scale = 5.0

    def read_sequence_info(self, seq_dir):
        p = read_json(os.path.join(seq_dir, "params.json"))
        parameters = np.array([p["initial_location"], p["contact_vector"]], dtype=np.float32).flatten()
        offsets = read_json(os.path.join(seq_dir, "metadata.json"))["rigid_pose_list"]
        return parameters, offsets

    def rest_mesh_path(self, seq_dir):
        name = os.path.basename(os.path.normpath(seq_dir))
        return os.path.join(seq_dir, f"{name}_1.ply")


class MassElasticDataset(DeformationSequenceDataset):
    """Soft ball dropped from 0.5 m (paper Sec. IV-A, Mass-Elastic).

    theta = (mass, bending stiffness) normalized to [-1, 1]; offset = height of the ball.
    """
    offset_dim = 1
    offset_scale = 2.0

    def __init__(self, *args, **kwargs):
        # The paper experiments keep the order of the split file for this dataset.
        kwargs.setdefault("sort_folders", False)
        super().__init__(*args, **kwargs)

    def read_sequence_info(self, seq_dir):
        p = read_json(os.path.join(seq_dir, "params.json"))
        parameters = normalize_mass_elastic_params(
            [p["mass"], p["soft_body_parameters"]["spring"]["bending"]])
        offsets = read_json(os.path.join(seq_dir, "metadata.json"))["offset_z_vertices_list"][-100:]
        return parameters, offsets

    def rest_mesh_path(self, seq_dir):
        name = os.path.basename(os.path.normpath(seq_dir))
        return os.path.join(seq_dir, f"{name}_0.ply")


DATASETS = {"contact_force": ContactForceDataset, "mass_elastic": MassElasticDataset}
