import contextlib
import logging
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

from .data import denormalize

METRICS = ("rmse", "mae", "mse")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def inference_mode(model, bn_batch_statistics=False):
    """``model.eval()`` + ``torch.no_grad()``.

    With ``bn_batch_statistics`` the BatchNorm layers normalize with the statistics of
    the current batch (how the paper numbers were computed) while their running
    statistics are left untouched. The previous state is restored on exit.
    """
    was_training = model.training
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    model.eval()
    if bn_batch_statistics:
        for m in bns:
            m.train()
            m.momentum = 0.0
    try:
        with torch.no_grad():
            yield model
    finally:
        for m, mom in zip(bns, momenta):
            m.momentum = mom
        model.train(was_training)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def setup_logger(log_file=None):
    logger = logging.getLogger("odeform")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    for h in handlers:
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
    return logger


# --------------------------------------------------------------------------------------
# Metrics (point-wise errors between corresponding vertices, in meters)
# --------------------------------------------------------------------------------------
def compute_metrics(pred, gt):
    diff = pred - gt
    mse = torch.mean(diff ** 2).item()
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": torch.mean(diff.abs()).item()}


class MetricAverager:
    def __init__(self):
        self.sums, self.count = defaultdict(float), 0

    def update(self, metrics):
        for k, v in metrics.items():
            self.sums[k] += v
        self.count += 1

    def average(self):
        return {k: v / max(self.count, 1) for k, v in self.sums.items()}


def format_metrics(m, unit="mm"):
    """RMSE/MAE in mm and MSE in mm^2 (as in the paper tables)."""
    if unit == "mm":
        return (f"RMSE {m['rmse'] * 1e3:.3f} mm | MAE {m['mae'] * 1e3:.3f} mm | "
                f"MSE {m['mse'] * 1e6:.3f} mm^2")
    return " | ".join(f"{k.upper()} {m[k]:.6f}" for k in METRICS)


def to_world(pcd_n, offset_n, norm, offset_scale, experiment):
    """Normalized prediction/GT -> world coordinates (meters).

    pcd_n:    (..., N, 3) normalized local shape.
    offset_n: (..., offset_dim) scaled global offset.
    """
    coord_min, coord_max, ref_mean = (torch.as_tensor(x, dtype=pcd_n.dtype, device=pcd_n.device)
                                      for x in norm)
    pcd = denormalize(pcd_n, coord_min, coord_max, ref_mean.reshape(-1))
    offset = offset_n / offset_scale
    if experiment == "mass_elastic":
        pcd = pcd.clone()
        pcd[..., 2] += offset[..., 0:1]  # height offset only
    else:
        pcd = pcd + offset.unsqueeze(-2)
    return pcd


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(path, model, awl, optimizer, epoch, loss, args):
    state = model.checkpoint_state()
    state.update({"awl_state_dict": awl.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch, "loss": loss, "args": vars(args)})
    torch.save(state, path)


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------
def save_prediction(vertices, reference_mesh, out_path):
    """Saves predicted vertices with the triangles of ``reference_mesh`` (or as a point
    cloud if the vertex counts differ or no mesh is available)."""
    import open3d as o3d

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vertices = vertices.detach().cpu().numpy().astype(np.float64)
    mesh = o3d.io.read_triangle_mesh(reference_mesh) if reference_mesh and os.path.exists(reference_mesh) else None
    if mesh is not None and len(mesh.vertices) == len(vertices) and len(mesh.triangles) > 0:
        out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), mesh.triangles)
        o3d.io.write_triangle_mesh(out_path, out)
    else:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        o3d.io.write_point_cloud(out_path, pcd)


def save_animation(frames, times, out_path, fps=10, gt_frames=None, triangles=None):
    """Writes a GIF of the predicted (and optionally ground-truth) sequence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    seqs = [("Prediction", [f.detach().cpu().numpy() for f in frames], "#6b8fd6")]
    if gt_frames is not None:
        seqs.append(("Ground truth", [f.detach().cpu().numpy() for f in gt_frames], "#6bbf73"))
    pts = np.concatenate([np.concatenate(s[1]) for s in seqs])
    mid, rng = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2

    fig = plt.figure(figsize=(5 * len(seqs), 5))
    axes = [fig.add_subplot(1, len(seqs), i + 1, projection="3d") for i in range(len(seqs))]

    def update(k):
        for ax, (title, s, color) in zip(axes, seqs):
            ax.clear()
            v = s[k]
            if triangles is not None and len(triangles):
                ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=triangles, color=color, alpha=0.85)
            else:
                ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c=color)
            for setlim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
                setlim(c - rng, c + rng)
            ax.set_box_aspect([1, 1, 1])
            ax.set_title(title)
        fig.suptitle(f"t = {float(times[k]):.2f} s")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer="pillow")
    plt.close(fig)
