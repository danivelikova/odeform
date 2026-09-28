"""Recover physical parameters from observed deformations (paper Sec. IV-G, Table V).

The trained ODeform model is frozen. For every test sequence of the Mass-Elastic dataset
one parameter (mass or bending stiffness) is replaced by a random initial guess in the
normalized range [-1, 1] and optimized by gradient descent (Eq. 10) on the reconstruction
error of the whole observed sequence. Gradients w.r.t. the parameter are computed with
central finite differences. The error is reported in the normalized parameter space
used by the model (paper) and in physical units.

Example:
    python optimize_parameters.py --config configs/mass_elastic/ball.yaml \
        --checkpoint checkpoints/mass_elastic/ball.pth --parameter both
"""
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from odeform.config import build_parser, parse_args
from odeform.data import DATASETS, get_splits
from odeform.models import ODeform
from odeform.utils import get_device, inference_mode, setup_logger

PARAMETERS = {"mass": 0, "bending": 1}
PHYSICAL_RANGE = 3.0  # both parameters were sampled in [0, 3]; normalized to [-1, 1]


def sequence_loss(model, offset, pcd, t, params):
    offset_pred, pcd_pred = model(offset[:, :1], pcd[:, 0], t, params)
    mse = nn.functional.mse_loss
    return (mse(offset_pred, offset) + mse(pcd_pred, pcd)).item()


def optimize(model, batch, device, index, lr, eps, patience, tol):
    offset, pcd = batch["offset"].to(device), batch["coords"].to(device)
    t, params = batch["time"][0].to(device), batch["parameters"].to(device)
    gt = params[0, index].item()

    params_opt = params.clone()
    params_opt[:, index] = torch.rand(1) * 2 - 1  # random initial guess in [-1, 1]
    init = params_opt[0, index].item()

    best_loss, best, no_improve = float("inf"), None, 0
    while True:
        loss = sequence_loss(model, offset, pcd, t, params_opt)
        plus, minus = params_opt.clone(), params_opt.clone()
        plus[:, index] += eps
        minus[:, index] -= eps
        grad = (sequence_loss(model, offset, pcd, t, plus) -
                sequence_loss(model, offset, pcd, t, minus)) / (2 * eps)
        params_opt[:, index] = (params_opt[:, index] - lr * grad).clamp(-1.0, 1.0)
        if loss < best_loss - tol:
            best_loss, best, no_improve = loss, params_opt[0, index].item(), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return dict(gt=gt, init=init, estimate=best, error=abs(best - gt), loss=best_loss)


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--parameter", choices=["mass", "bending", "both"], default="both")
    parser.add_argument("--opt_lr", type=float, default=10.0)
    parser.add_argument("--fd_eps", type=float, default=1e-4)
    parser.add_argument("--opt_patience", type=int, default=10)
    parser.add_argument("--opt_tol", type=float, default=1e-8)
    parser.add_argument("--init_seed", type=int, default=23,
                        help="seed for the random initial guesses (23 reproduces Table V)")
    args = parse_args(parser)
    if args.experiment != "mass_elastic":
        raise SystemExit("Parameter optimization is defined for the mass_elastic experiment.")
    device = get_device(args.device)
    logger = setup_logger(os.path.join(args.output_dir, "optimize_parameters.log"))

    _, _, test_folders = get_splits(args.dataset_path)
    dataset = DATASETS[args.experiment](
        args.dataset_path, test_folders, seq_start_idx=args.seq_start_idx,
        sequence_length=args.sequence_length, fps=args.fps)

    names = list(PARAMETERS) if args.parameter == "both" else [args.parameter]
    summary = {}
    for name in names:
        # Same random-number sequence as the original experiments: seed, build the
        # model (weight init consumes random numbers), then draw one guess per sequence.
        torch.manual_seed(args.init_seed)
        model = ODeform(args.experiment, args.deform_encoder, args.latent_dim,
                        args.ode_hidden_layers)
        model.load_checkpoint(args.checkpoint)
        model.to(device)

        errors = []
        # Iterating a DataLoader also draws from the global RNG (kept for reproducibility).
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with inference_mode(model, args.bn_batch_statistics):
            for batch in loader:
                r = optimize(model, batch, device, PARAMETERS[name], args.opt_lr,
                             args.fd_eps, args.opt_patience, args.opt_tol)
                errors.append(r["error"])
                logger.info(f"[{name}] {batch['name'][0]}: gt {r['gt']:+.4f} "
                            f"init {r['init']:+.4f} -> estimate {r['estimate']:+.4f} "
                            f"(error {r['error']:.4f})")
        mae = sum(errors) / len(errors)
        summary[name] = {"mae_normalized": mae, "mae_physical": mae * PHYSICAL_RANGE / 2,
                         "num_sequences": len(errors)}
        logger.info(f"[{name}] MAE over {len(errors)} sequences: {mae:.3f} (normalized), "
                    f"{mae * PHYSICAL_RANGE / 2:.3f} (physical units)")

    with open(os.path.join(args.output_dir, "optimize_parameters.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
