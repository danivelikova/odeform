"""Train ODeform on the Contact-Force or Mass-Elastic dataset.

Example:
    python train.py --config configs/contact_force/donut.yaml
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from odeform.config import build_parser, parse_args
from odeform.data import DATASETS, get_splits
from odeform.models import AutomaticWeightedLoss, ODeform
from odeform.utils import (get_device, inference_mode, save_checkpoint, seed_everything,
                           setup_logger)


def forward_pass(model, batch, device, criterion, awl):
    pcd = batch["coords"].to(device)          # (B, T, N, 3)
    offset = batch["offset"].to(device)       # (B, T, offset_dim)
    params = batch["parameters"].to(device)   # (B, P)
    t = batch["time"][0].to(device)           # (T,)
    offset_pred, pcd_pred = model(offset[:, :1], pcd[:, 0], t, params)
    offset_loss = criterion(offset_pred, offset)
    deform_loss = criterion(pcd_pred, pcd)
    return awl(offset_loss, deform_loss), offset_loss, deform_loss


def main():
    args = parse_args(build_parser(__doc__))
    seed_everything(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, "train.log"))
    logger.info(f"Arguments: {vars(args)}")

    train_folders, val_folders, _ = get_splits(args.dataset_path)
    Dataset = DATASETS[args.experiment]
    kw = dict(seq_start_idx=args.seq_start_idx, sequence_length=args.sequence_length,
              fps=args.fps, frame_interval=args.frame_interval)
    train_set = Dataset(args.dataset_path, train_folders, **kw)
    val_set = Dataset(args.dataset_path, val_folders, **kw)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = ODeform(args.experiment, args.deform_encoder, args.latent_dim,
                    args.ode_hidden_layers).to(device)
    awl = AutomaticWeightedLoss(2).to(device)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint, map_location=device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "weight_decay": args.weight_decay},
        {"params": awl.parameters(), "weight_decay": 0.0},
    ], lr=args.lr)
    criterion = nn.MSELoss()
    logger.info(model)

    best_loss, best_epoch, bad_epochs = float("inf"), -1, 0
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            loss, _, _ = forward_pass(model, batch, device, criterion, awl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss, val_offset, val_deform = 0.0, 0.0, 0.0
        with inference_mode(model, args.bn_batch_statistics):
            for batch in val_loader:
                loss, o, d = forward_pass(model, batch, device, criterion, awl)
                val_loss, val_offset, val_deform = val_loss + loss.item(), val_offset + o.item(), val_deform + d.item()
        n = len(val_loader)
        val_loss, val_offset, val_deform = val_loss / n, val_offset / n, val_deform / n
        logger.info(f"Epoch {epoch:4d} | train {train_loss:.6f} | val {val_loss:.6f} "
                    f"(offset {val_offset:.6f}, deform {val_deform:.6f})")

        if epoch < args.min_epochs:
            continue
        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            save_checkpoint(os.path.join(args.output_dir, "best.pth"), model, awl, optimizer,
                            epoch, best_loss, args)
            logger.info(f"  saved best model (epoch {epoch}, val {best_loss:.6f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}; best epoch {best_epoch} "
                            f"(val {best_loss:.6f})")
                break

    save_checkpoint(os.path.join(args.output_dir, "last.pth"), model, awl, optimizer,
                    epoch, val_loss, args)
    logger.info("Finished training")


if __name__ == "__main__":
    main()
