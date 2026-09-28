"""Command line / YAML configuration shared by all entry points.

Values are resolved as: built-in defaults < YAML file (``--config``) < command line.
"""
import argparse

import yaml


def build_parser(description=""):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML config file")

    g = p.add_argument_group("experiment")
    g.add_argument("--experiment", choices=["contact_force", "mass_elastic"], default="contact_force")
    g.add_argument("--dataset_path", type=str, default=None)
    g.add_argument("--output_dir", type=str, default="outputs/run")
    g.add_argument("--checkpoint", type=str, default=None, help="checkpoint to evaluate / fine-tune")
    g.add_argument("--device", type=str, default="auto")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("data")
    g.add_argument("--seq_start_idx", type=int, default=0, help="first frame of the window")
    g.add_argument("--sequence_length", type=int, default=10, help="number of frames in the window")
    g.add_argument("--fps", type=int, default=10)
    g.add_argument("--frame_interval", type=int, default=1,
                   help="train on every k-th frame only (sparse supervision)")
    g.add_argument("--batch_size", type=int, default=64)

    g = p.add_argument_group("model")
    g.add_argument("--deform_encoder", choices=["mlp", "pemlp", "pointnet", "pointnet_bn"],
                   default="pointnet_bn")
    g.add_argument("--latent_dim", type=int, default=128)
    g.add_argument("--ode_hidden_layers", type=int, default=5)

    g = p.add_argument_group("training")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-4)
    g.add_argument("--num_epochs", type=int, default=5000)
    g.add_argument("--min_epochs", type=int, default=100)
    g.add_argument("--early_stopping_patience", type=int, default=20)

    g = p.add_argument_group("evaluation")
    g.add_argument("--split", choices=["train", "val", "test"], default="test")
    g.add_argument("--bn_batch_statistics", type=_bool, default=True,
                   help="normalize the PointNet BatchNorm layers with the statistics of the "
                        "evaluation batch (as done for the paper numbers) instead of the "
                        "running statistics")
    g.add_argument("--save_meshes", action="store_true", help="export predicted/GT meshes")
    g.add_argument("--save_animations", action="store_true", help="export a GIF per sequence")
    return p


def _bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args(parser):
    pre, _ = parser.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown keys in {pre.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args()
