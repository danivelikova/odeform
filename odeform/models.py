"""ODeform model: two parallel latent neural ODEs.

* ``GlobalMotionODE``  (paper: E_g, f_g, D_g) models the rigid/global motion of the
  object, i.e. the per-frame translation of the object (``offset``), with a 32-d latent.
* ``DeformationODE``   (paper: E_l, f_l, D_l) models the per-point local deformation
  with a 128-d latent per point. Its vector field is conditioned on the global latent.

Attribute names (``encoder``, ``ode_func``, ``decoder`` and their sub-layers) are kept
identical to the original research code so that the released checkpoints load
without key remapping.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq

GLOBAL_LATENT_DIM = 32
GLOBAL_ODE_HIDDEN_LAYERS = 5


# --------------------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------------------
def _concat_parameters(x, parameters):
    """Broadcast physical parameters (B, P) to every point of x (B, N, D) and concat."""
    parameters_expanded = parameters.unsqueeze(1).expand(-1, x.shape[1], -1)
    return torch.cat([x, parameters_expanded], dim=-1)


def positional_encoding(x, num_encoding_functions=6, include_input=True):
    encoding = [x] if include_input else []
    frequency_bands = 2.0 ** torch.linspace(0.0, num_encoding_functions - 1,
                                            num_encoding_functions, device=x.device)
    for freq in frequency_bands:
        for func in (torch.sin, torch.cos):
            encoding.append(func(x * freq * math.pi))
    return torch.cat(encoding, dim=-1)


class EncoderMLP(nn.Module):
    """Point-wise MLP on [x, theta]."""

    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x, parameters):
        return self.encoder_net(_concat_parameters(x, parameters))


class EncoderPeMLP(nn.Module):
    """Point-wise MLP on [PE(xyz), theta] (used for the Mass-Elastic dataset)."""

    def __init__(self, input_dim, latent_dim=64, num_encoding_functions=6, include_input=True):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        pe_input_dim = 3 * (2 * num_encoding_functions) + input_dim
        self.encoder_net = nn.Sequential(
            nn.Linear(pe_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x, parameters):
        x_pe = positional_encoding(x, self.num_encoding_functions, self.include_input)
        return self.encoder_net(_concat_parameters(x_pe, parameters))


class STN3d(nn.Module):
    """PointNet input transform (adapted from fxia22/pointnet.pytorch)."""

    def __init__(self, input_dim=3, batch_norm=False):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
        self.input_dim = input_dim

    def _bn(self, name, x):
        return getattr(self, name)(x) if self.batch_norm else x

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self._bn("bn1", self.conv1(x)))
        x = F.relu(self._bn("bn2", self.conv2(x)))
        x = F.relu(self._bn("bn3", self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0].view(-1, 1024)
        x = F.relu(self._bn("bn4", self.fc1(x)))
        x = F.relu(self._bn("bn5", self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.input_dim, device=x.device).view(1, -1).repeat(batch_size, 1)
        return (x + iden).view(-1, self.input_dim, self.input_dim)


class EncoderPointNet(nn.Module):
    """Point-wise PointNet features on [x, theta] (used for the Contact-Force dataset).

    ``batch_norm=True`` matches the released Contact-Force checkpoints (Tables I/II);
    ``batch_norm=False`` matches the checkpoint used for the HouseCat6D demo.
    """

    def __init__(self, input_dim, latent_dim=64, batch_norm=True):
        super().__init__()
        self.stn = STN3d(input_dim=3, batch_norm=batch_norm)
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_dim, 1)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(latent_dim)

    def _bn(self, name, x):
        return getattr(self, name)(x) if self.batch_norm else x

    def forward(self, x, parameters):
        x = _concat_parameters(x, parameters).transpose(2, 1)  # (B, D, N)
        # Spatial transformer on the xyz channels only.
        trans = self.stn(x[:, :3, :])
        xyz = torch.bmm(x[:, :3, :].transpose(2, 1), trans).transpose(2, 1)
        x = torch.cat([xyz, x[:, 3:, :]], dim=1)
        x = F.relu(self._bn("bn1", self.conv1(x)))
        x = F.relu(self._bn("bn2", self.conv2(x)))
        x = self._bn("bn3", self.conv3(x))
        return x.transpose(2, 1)  # (B, N, latent_dim)


ENCODERS = {
    "mlp": lambda i, l: EncoderMLP(i, l),
    "pemlp": lambda i, l: EncoderPeMLP(i, l),
    "pointnet": lambda i, l: EncoderPointNet(i, l, batch_norm=False),
    "pointnet_bn": lambda i, l: EncoderPointNet(i, l, batch_norm=True),
}


# --------------------------------------------------------------------------------------
# ODE vector field and decoder
# --------------------------------------------------------------------------------------
class ODEFunc(nn.Module):
    """MLP vector field f(z, t) (optionally also conditioned on the global latent)."""

    def __init__(self, latent_dim=64, ode_hidden_layers=1, cond_dim=0):
        super().__init__()
        layers = [nn.Linear(latent_dim + cond_dim + 1, 2 * latent_dim), nn.Tanh()]
        for _ in range(ode_hidden_layers - 1):
            layers += [nn.Linear(2 * latent_dim, 2 * latent_dim), nn.Tanh()]
        layers.append(nn.Linear(2 * latent_dim, latent_dim))
        self.mlp_net = nn.Sequential(*layers)
        self.cond_dim = cond_dim
        # Conditioning tensor (B, 1, cond_dim), set by ODeform before integrating.
        self.z_cond = None

    def forward(self, t, z):
        # z: (B, N, latent_dim)
        if self.cond_dim:
            z = torch.cat((z, self.z_cond.expand(-1, z.size(1), -1)), dim=2)
        t_input = t * torch.ones_like(z[:, :, :1])
        return self.mlp_net(torch.cat([z, t_input], dim=2))


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim=64):
        super().__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2), nn.ReLU(),
            nn.Linear(latent_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.decoder_net(x)


class LatentODE(nn.Module):
    """Encoder -> latent neural ODE (adjoint, RK4) -> point-wise decoder."""

    def __init__(self, encoder, latent_dim, ode_hidden_layers, input_dim_enc, output_dim_dec,
                 cond_dim=0):
        super().__init__()
        self.encoder = ENCODERS[encoder](input_dim_enc, latent_dim)
        self.ode_func = ODEFunc(latent_dim, ode_hidden_layers, cond_dim=cond_dim)
        self.decoder = Decoder(output_dim_dec, latent_dim)

    def integrate(self, x0, t_sequence, parameters):
        """Returns the latent trajectory (B, T, N, latent_dim)."""
        z0 = self.encoder(x0, parameters)
        z_t = torchdiffeq.odeint_adjoint(self.ode_func, z0, t_sequence, method="rk4",
                                         adjoint_options=dict(norm="seminorm"))
        return z_t.permute(1, 0, 2, 3)

    def forward(self, x0, t_sequence, parameters):
        z_t = self.integrate(x0, t_sequence, parameters)
        return self.decoder(z_t), z_t  # (B, T, N, out_dim)


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------
EXPERIMENTS = {
    # Contact-Force: theta = (contact point xyz, force vector xyz); offset = rigid xyz.
    "contact_force": dict(param_dim=6, offset_dim=3, offset_scale=5.0),
    # Mass-Elastic: theta = (mass, bending stiffness); offset = height (z) of the object.
    "mass_elastic": dict(param_dim=2, offset_dim=1, offset_scale=2.0),
}


class ODeform(nn.Module):
    """Global-motion ODE + deformation ODE (Fig. 2 of the paper).

    Inputs:
        offset0:    (B, 1, offset_dim) initial global offset of the object (scaled).
        pcd0:       (B, N, 3)          initial normalized point cloud.
        t_sequence: (T,)               query times.
        parameters: (B, P)             physical parameters theta.
    Returns:
        offset_pred (B, T, offset_dim), pcd_pred (B, T, N, 3)
    """

    def __init__(self, experiment, deform_encoder, latent_dim=128, ode_hidden_layers=5):
        super().__init__()
        cfg = EXPERIMENTS[experiment]
        self.experiment = experiment
        self.offset_scale = cfg["offset_scale"]
        p = cfg["param_dim"]
        self.global_ode = LatentODE("mlp", GLOBAL_LATENT_DIM, GLOBAL_ODE_HIDDEN_LAYERS,
                                    input_dim_enc=cfg["offset_dim"] + p,
                                    output_dim_dec=cfg["offset_dim"])
        self.deform_ode = LatentODE(deform_encoder, latent_dim, ode_hidden_layers,
                                    input_dim_enc=3 + p, output_dim_dec=3,
                                    cond_dim=GLOBAL_LATENT_DIM)

    def forward(self, offset0, pcd0, t_sequence, parameters):
        offset_pred, z_global = self.global_ode(offset0, t_sequence, parameters)
        # The deformation field is conditioned on the global latent. As in the
        # experiments of the paper, the global latent of the second time step is
        # used (constant over the integration interval).
        self.deform_ode.ode_func.z_cond = z_global[:, 1]
        pcd_pred, _ = self.deform_ode(pcd0, t_sequence, parameters)
        return offset_pred.squeeze(2), pcd_pred

    # Checkpoints use the key layout of the original code base.
    def load_checkpoint(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.global_ode.load_state_dict(ckpt["model1_state_dict"])
        self.deform_ode.load_state_dict(ckpt["model2_state_dict"])
        return ckpt

    def checkpoint_state(self):
        return {"model1_state_dict": self.global_ode.state_dict(),
                "model2_state_dict": self.deform_ode.state_dict()}


class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss (Liebel & Koerner, 2018), Eq. (9)."""

    def __init__(self, num=2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))

    def forward(self, *losses):
        total = 0
        for i, loss in enumerate(losses):
            total = total + 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return total
