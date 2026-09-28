"""Predict the deformation of a single (possibly unseen) shape.

The shape is normalized with its own bounding box, deformed with a trained model for the
given physical parameters, and the predicted meshes (or point clouds) are written for
every query time. Because the dynamics are continuous, the output frame rate
(``--fps``) and number of frames (``--num_frames``) can differ from training.

Contact-Force example (HouseCat6D shoe, paper Fig. 5):
    python infer.py --config configs/demo/housecat6d_shoe.yaml \
        --input data/housecat6d/aligned_mesh_shoe-cat_grey_sandal_right.ply

Mass-Elastic example:
    python infer.py --config configs/mass_elastic/ball.yaml \
        --checkpoint checkpoints/mass_elastic/ball.pth \
        --input data/mass_elastic/<seq>/<seq>_0.ply --mass 1.0 --bending 0.5 --initial_offset 0.3
"""
import os

import numpy as np
import torch

from odeform.config import build_parser, parse_args
from odeform.data import load_points, normalization_params, normalize, normalize_mass_elastic_params
from odeform.models import EXPERIMENTS, ODeform
from odeform.utils import get_device, inference_mode, save_animation, save_prediction, setup_logger, to_world


def main():
    parser = build_parser(__doc__)
    g = parser.add_argument_group("inference")
    g.add_argument("--input", type=str, default=None, help="input mesh (.ply/.obj) or point cloud (.xyz)")
    g.add_argument("--num_frames", type=int, default=None, help="default: --sequence_length")
    g.add_argument("--t0", type=float, default=None, help="start time; default: seq_start_idx / fps")
    g.add_argument("--contact_point", type=float, nargs=3, default=None, help="[contact_force] xyz")
    g.add_argument("--contact_vector", type=float, nargs=3, default=None, help="[contact_force] xyz")
    g.add_argument("--mass", type=float, default=None, help="[mass_elastic] kg, in [0, 3]")
    g.add_argument("--bending", type=float, default=None, help="[mass_elastic] in [0, 3]")
    g.add_argument("--initial_offset", type=float, nargs="+", default=None,
                   help="initial global offset: xyz (contact_force) or height z (mass_elastic)")
    args = parse_args(parser)
    if not (args.checkpoint and args.input):
        raise SystemExit("--checkpoint and --input are required")
    device = get_device(args.device)
    logger = setup_logger(os.path.join(args.output_dir, "infer.log"))

    exp = EXPERIMENTS[args.experiment]
    if args.experiment == "contact_force":
        if args.contact_point is None or args.contact_vector is None:
            raise SystemExit("--contact_point and --contact_vector are required")
        params = np.array(args.contact_point + args.contact_vector, dtype=np.float32)
    else:
        if args.mass is None or args.bending is None:
            raise SystemExit("--mass and --bending are required")
        params = normalize_mass_elastic_params([args.mass, args.bending])
    offset0 = np.zeros(exp["offset_dim"], np.float32) if args.initial_offset is None \
        else np.asarray(args.initial_offset, np.float32)
    if offset0.shape != (exp["offset_dim"],):
        raise SystemExit(f"--initial_offset needs {exp['offset_dim']} value(s)")

    points = load_points(args.input)
    norm = normalization_params(points)
    pcd0 = torch.from_numpy(normalize(points, *norm)).float()[None].to(device)
    offset0 = torch.from_numpy(exp["offset_scale"] * offset0)[None, None].to(device)
    params = torch.from_numpy(params)[None].to(device)
    num_frames = args.num_frames or args.sequence_length
    t0 = args.seq_start_idx / args.fps if args.t0 is None else args.t0
    t = t0 + torch.arange(num_frames, dtype=torch.float32, device=device) / args.fps

    model = ODeform(args.experiment, args.deform_encoder, args.latent_dim,
                    args.ode_hidden_layers).to(device)
    model.load_checkpoint(args.checkpoint, map_location=device)
    with inference_mode(model, args.bn_batch_statistics):
        offset_pred, pcd_pred = model(offset0, pcd0, t, params)
    pred = to_world(pcd_pred, offset_pred, norm, exp["offset_scale"], args.experiment)[0].cpu()

    name = os.path.splitext(os.path.basename(args.input))[0]
    for k in range(num_frames):
        out = os.path.join(args.output_dir, name, f"{name}_t{k:03d}.ply")
        save_prediction(pred[k], args.input if not args.input.endswith(".xyz") else None, out)
    logger.info(f"Wrote {num_frames} frames (t = {t[0]:.2f} .. {t[-1]:.2f} s) to "
                f"{os.path.join(args.output_dir, name)}")
    if args.save_animations:
        tris = None
        if not args.input.endswith(".xyz"):
            import open3d as o3d
            tris = np.asarray(o3d.io.read_triangle_mesh(args.input).triangles)
        save_animation(list(pred), t.cpu(), os.path.join(args.output_dir, f"{name}.gif"),
                       fps=args.fps, triangles=tris)


if __name__ == "__main__":
    main()
