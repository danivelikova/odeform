# ODeform: Learning Continuous 4D Motion for Shape Deformation with Neural ODEs

[Yordanka Velikova](https://github.com/danivelikova), Mahdi Saleh, Liming Kuang, Benjamin Busam
Technical University of Munich · Munich Center for Machine Learning

[[Paper (arXiv:2607.20670)]](https://arxiv.org/abs/2607.20670)

ODeform encodes an initial point cloud and its physical parameters (e.g. contact force,
mass, bending stiffness) into a latent space and evolves it continuously in time with two
parallel neural ODEs: one for the global (rigid) motion of the object and one for the local
deformation. Decoding the latent trajectories gives the deformed shape at any query time.

This repository contains the code, pretrained models and data for the **Contact-Force** and
**Mass-Elastic** experiments of the paper.

| Paper | Code |
|---|---|
| Dual encoder E_g (32-d), E_l (128-d) — Eq. (5) | `odeform/models.py`: `ODeform.global_ode.encoder`, `ODeform.deform_ode.encoder` |
| Parallel neural ODEs f_g, f_l — Eqs. (6), (7) | `ODEFunc`, integrated with `torchdiffeq.odeint_adjoint` (RK4) |
| Decoders D_g, D_l — Eq. (8) | `Decoder` |
| Automatically weighted loss — Eq. (9) | `AutomaticWeightedLoss` |
| Datasets — Sec. IV-A | `odeform/data.py` |
| Parameter optimization — Eq. (10), Sec. IV-G | `optimize_parameters.py` |

## Installation

```bash
git clone https://github.com/danivelikova/odeform.git
cd odeform
conda env create -f environment.yml
conda activate odeform
```

Or, with an existing PyTorch (≥ 2.0) installation: `pip install -r requirements.txt`.
The code was tested with Python 3.11, PyTorch 2.1.1 and CUDA 12.1 and also runs on CPU.

## Pretrained models and data

All assets are hosted in [this Google Drive folder](https://drive.google.com/drive/folders/1FmD-MZsBkA3cIzFlTwfNlf-u1JqdCRyy).
The download script fetches and unpacks them into `checkpoints/` and `data/`:

```bash
bash scripts/download_assets.sh                     # pretrained models only (39 MB)
bash scripts/download_assets.sh checkpoints donut   # models + one Contact-Force object
bash scripts/download_assets.sh all                 # models + all datasets (~3.5 GB)
```

| Asset | File | Content |
|---|---|---|
| `checkpoints` | `odeform_checkpoints.zip` | one model per Contact-Force object, the Mass-Elastic model, the HouseCat6D demo model |
| `bottle` … `pillow` | `contact_force_<object>.zip` | 300 sequences per object + train/val/test split |
| `mass_elastic` | `mass_elastic.zip` | 500 drop sequences + train/val/test split |

After downloading, the layout is:

```
checkpoints/
  contact_force/{bottle,cat,dog,donut,doritos,flipflop,pillow}.pth
  mass_elastic/ball.pth
  demo/flipflop_nobn.pth
data/
  contact_force/<object>/{train,val,test}_folders.txt
  contact_force/<object>/<sequence>/...
  mass_elastic/{train,val,test}_folders.txt
  mass_elastic/<sequence>/...
```

### Data format

Every sequence folder contains

| File | Content |
|---|---|
| `params.json` | physical parameters. Contact-Force: `initial_location` (contact point) and `contact_vector` (force direction/magnitude). Mass-Elastic: `mass` and `soft_body_parameters.spring.bending` |
| `metadata.json` | per-frame global motion. Contact-Force: `rigid_pose_list` (xyz translation). Mass-Elastic: `offset_z_vertices_list` (height of the ball) |
| `xyz_meshes/<seq>_<k>.xyz` | vertex positions (and normals) of frame `k` |
| `<seq>_<k>.ply` | mesh of frame `k` (vertex order identical to the `.xyz` file) |

* **Contact-Force** (paper Sec. IV-A): seven soft objects from the Everyday Deform
  dataset hit by rigid projectiles with random contact points and force vectors, simulated
  in Blender. 7 × 300 sequences, 10 fps. Models are trained and evaluated on the first 10
  frames. Split 70/10/20 (210/30/60) per object.
* **Mass-Elastic** (paper Sec. IV-A): a soft ball dropped from 0.5 m with random mass and
  bending stiffness. 500 sequences of 100 frames at 20 fps. Models are trained and evaluated
  on the 20 frames around the ground impact (frames 40–59, t = 2.0–2.95 s).
  Split 350/50/100. Both parameters are mapped from the range [0, 3] to [-1, 1].

The provided `*_folders.txt` files are the splits used in the paper. If they are missing,
`odeform.data.get_splits` regenerates the same split (fixed seed).

## Evaluation

Contact-Force, all objects (Table II; the average is the Contact-Force column of Table I):

```bash
bash scripts/eval_contact_force.sh              # add --device cpu to run on CPU
```

Single object, optionally exporting meshes and GIF animations of prediction vs. ground truth:

```bash
python evaluate.py --config configs/contact_force/donut.yaml \
    --checkpoint checkpoints/contact_force/donut.pth --save_meshes --save_animations
```

Mass-Elastic (Table I):

```bash
python evaluate.py --config configs/mass_elastic/ball.yaml --checkpoint checkpoints/mass_elastic/ball.pth
```

Metrics are computed between corresponding vertices of the predicted and ground-truth
shapes in world coordinates (RMSE, MAE in mm; MSE in mm²), per frame, and averaged over
all frames and test sequences. Results are written to `outputs/.../metrics_test.json`.

### Parameter optimization (Table V)

Recover the mass or bending stiffness of each Mass-Elastic test sequence from the observed
deformation with the frozen model (Eq. 10):

```bash
python optimize_parameters.py --config configs/mass_elastic/ball.yaml \
    --checkpoint checkpoints/mass_elastic/ball.pth --parameter both --device cpu
```

The reported MAE is measured in the normalized parameter space of the model ([-1, 1]),
as in the paper; the value in physical units is logged as well.

### Unseen real shapes: HouseCat6D (Fig. 5)

1. Download the shoe meshes of [HouseCat6D](https://sites.google.com/view/housecat6d)
   (`obj_models/shoe`).
2. Align them to the flip-flop of the Contact-Force dataset:
   ```bash
   python scripts/align_housecat6d.py --meshes /path/to/housecat6d/obj_models/shoe \
       --reference data/contact_force/flipflop/1112_224643/1112_224643_1.ply --output data/housecat6d
   ```
3. Deform them with the flip-flop model:
   ```bash
   python infer.py --config configs/demo/housecat6d_shoe.yaml \
       --input data/housecat6d/aligned_mesh_shoe-cat_grey_sandal_right.ply --save_animations
   ```

`infer.py` works for any input mesh/point cloud and physical parameters (see
`python infer.py -h`). Since the dynamics are continuous, the output can be queried at a
different frame rate or number of frames than used for training (`--fps`, `--num_frames`).

## Training

```bash
python train.py --config configs/contact_force/donut.yaml     # one model per object
python train.py --config configs/mass_elastic/ball.yaml
```

Any config value can be overridden on the command line (e.g. `--batch_size 32
--output_dir outputs/my_run`). The best model on the validation split is stored as
`<output_dir>/best.pth`. `--frame_interval k` trains with only every k-th frame of the
window (sparse supervision) while evaluation still uses all frames.

## Reproduced results

Numbers obtained with this code and the released checkpoints (CPU, test split):

| Contact-Force (Table II) | RMSE (mm) | MAE (mm) | MSE (mm²) |
|---|---:|---:|---:|
| Bottle | 7.322 | 5.733 | 64.7 |
| Cat | 9.298 | 7.203 | 107.5 |
| Dog | 10.617 | 8.380 | 127.9 |
| Donut | 4.058 | 3.000 | 18.1 |
| Doritos | 4.842 | 3.676 | 26.3 |
| Flipflop | 5.760 | 4.428 | 38.5 |
| Pillow | 4.705 | 3.560 | 25.3 |
| **Average** | **6.657** | **5.140** | **58.3** |

| Mass-Elastic (Table I) | RMSE (mm) | MAE (mm) | MSE (mm²) |
|---|---:|---:|---:|
| ODeform | 1.301 | 0.930 | 2.67 |

| Parameter optimization (Table V) | Mass | Bending |
|---|---:|---:|
| MAE (normalized parameter space) | 0.113 | 0.083 |

## Implementation notes

* **BatchNorm at test time.** The Contact-Force models use a PointNet encoder with
  BatchNorm. As in the paper experiments, BatchNorm normalizes with the statistics of the
  evaluation batch (`--bn_batch_statistics true`, the default). Results therefore depend on
  the batch composition; use the default `batch_size: 64` (the whole test split of an
  object) to reproduce the numbers above.
* **Global-latent conditioning.** The deformation ODE is conditioned on the global latent
  of the second query time, which is kept constant during integration.
* **Normalization.** All shapes of a split are normalized with the first frame of the first
  sequence of that split. Predictions are mapped back to world coordinates before
  computing metrics.

## Citation

```bibtex
@article{velikova2026odeform,
  title   = {ODeform: Learning Continuous 4D Motion for Shape Deformation with Neural ODEs},
  author  = {Velikova, Yordanka and Saleh, Mahdi and Kuang, Liming and Busam, Benjamin},
  journal = {arXiv preprint arXiv:2607.20670},
  year    = {2026}
}
```

## Acknowledgements

The Contact-Force dataset extends the Everyday Deform dataset of
[Saleh et al., ICRA 2024](https://arxiv.org/abs/2402.03466). The PointNet encoder is adapted
from [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) and the ODEs are solved
with [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

## License

The code is released under the [MIT License](LICENSE).
