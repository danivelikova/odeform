"""Evaluate an ODeform checkpoint on a dataset split (paper Tables I and II).

Metrics are computed between corresponding vertices of the predicted and ground-truth
shapes in world coordinates, per frame, and averaged over all frames and sequences.

Example:
    python evaluate.py --config configs/contact_force/donut.yaml \
        --checkpoint checkpoints/contact_force/donut.pth
"""
import json
import os

import numpy as np
from torch.utils.data import DataLoader

from odeform.config import build_parser, parse_args
from odeform.data import DATASETS, get_splits
from odeform.models import ODeform
from odeform.utils import (MetricAverager, compute_metrics, format_metrics, get_device,
                           inference_mode, save_animation, save_prediction, seed_everything,
                           setup_logger, to_world)


def main():
    args = parse_args(build_parser(__doc__))
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")
    seed_everything(args.seed)
    device = get_device(args.device)
    logger = setup_logger(os.path.join(args.output_dir, f"eval_{args.split}.log"))

    splits = dict(zip(("train", "val", "test"), get_splits(args.dataset_path)))
    dataset = DATASETS[args.experiment](
        args.dataset_path, splits[args.split], seq_start_idx=args.seq_start_idx,
        sequence_length=args.sequence_length, fps=args.fps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = ODeform(args.experiment, args.deform_encoder, args.latent_dim,
                    args.ode_hidden_layers).to(device)
    ckpt = model.load_checkpoint(args.checkpoint, map_location=device)
    logger.info(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    overall, per_frame, per_sequence = MetricAverager(), {}, {}
    with inference_mode(model, args.bn_batch_statistics):
        for batch in loader:
            pcd, offset = batch["coords"].to(device), batch["offset"].to(device)
            params, t = batch["parameters"].to(device), batch["time"][0].to(device)
            offset_pred, pcd_pred = model(offset[:, :1], pcd[:, 0], t, params)

            pred = to_world(pcd_pred, offset_pred, dataset.norm, model.offset_scale, args.experiment).cpu()
            gt = to_world(pcd, offset, dataset.norm, model.offset_scale, args.experiment).cpu()

            for b, name in enumerate(batch["name"]):
                seq_avg = MetricAverager()
                for k in range(len(t)):
                    m = compute_metrics(pred[b, k], gt[b, k])
                    overall.update(m)
                    seq_avg.update(m)
                    per_frame.setdefault(k, MetricAverager()).update(m)
                per_sequence[name] = seq_avg.average()
                logger.info(f"{name}: {format_metrics(per_sequence[name])}")

                mesh = batch["mesh"][b]
                if args.save_meshes:
                    for k in range(len(t)):
                        save_prediction(pred[b, k], mesh, os.path.join(args.output_dir, "pred", name, f"{k:03d}.ply"))
                        save_prediction(gt[b, k], mesh, os.path.join(args.output_dir, "gt", name, f"{k:03d}.ply"))
                if args.save_animations:
                    tris = None
                    if mesh and os.path.exists(mesh):
                        import open3d as o3d
                        tris = np.asarray(o3d.io.read_triangle_mesh(mesh).triangles)
                    save_animation(list(pred[b]), t.cpu(), os.path.join(args.output_dir, "animations", f"{name}.gif"),
                                   fps=args.fps, gt_frames=list(gt[b]), triangles=tris)

    results = overall.average()
    logger.info("Per-frame averages:")
    for k in sorted(per_frame):
        logger.info(f"  frame {dataset.frame_ids[k]:3d} (t={float(t[k]):.2f}s): {format_metrics(per_frame[k].average())}")
    logger.info(f"Overall ({len(dataset)} sequences x {len(t)} frames): {format_metrics(results)}")

    with open(os.path.join(args.output_dir, f"metrics_{args.split}.json"), "w") as f:
        json.dump({"overall_m": results,
                   "overall_mm": {"rmse": results["rmse"] * 1e3, "mae": results["mae"] * 1e3,
                                  "mse": results["mse"] * 1e6},
                   "per_sequence_m": per_sequence}, f, indent=2)


if __name__ == "__main__":
    main()
