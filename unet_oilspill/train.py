"""Entrenamiento U-Net + ResNet34 (transfer learning) para OCEAN GUARD.

Fase 1: encoder congelado (BatchNorm del encoder en modo eval), solo decoder.
Fase 2: fine-tuning completo con learning rates diferenciados (encoder y decoder).

Entrenamiento: recortes aleatorios de 256x256, batch 2.
Validacion y test: imagen completa con padding a multiplo de 32, batch 1.
Test solo se usa con --evaluate-test, una vez, con el mejor checkpoint.

Uso:
  python train.py --out runs\\unet_resnet34_v1
  python train.py --out runs\\unet_resnet34_v1 --evaluate-test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

from data import (
    CROP_SIZE,
    DEFAULT_DATASET_ROOT,
    MULTIPLE,
    crop_pair,
    load_image,
    load_mask_binary,
    pad_amounts,
    pad_hw,
    random_crop_coords,
    read_split,
)
from metrics import Confusion
from model import (
    WEIGHTS_PATH,
    build_model,
    count_trainable,
    decoder_parameters,
    encoder_parameters,
    set_encoder_trainable,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainDataset(Dataset):
    """Un recorte aleatorio 256x256 por muestra y epoca.

    El generador depende de (seed, epoca, indice): el mismo recorte se reproduce
    independientemente del numero de workers.
    """

    def __init__(self, samples, seed: int) -> None:
        self.samples = samples
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = load_image(s.image_path)
        mask = load_mask_binary(s.mask_path)
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        top, left = random_crop_coords(mask.shape[0], mask.shape[1], CROP_SIZE, rng)
        img_c, msk_c = crop_pair(img, mask, top, left, CROP_SIZE)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img_c)).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(np.ascontiguousarray(msk_c)).unsqueeze(0).contiguous(),
        }


class EvalDataset(Dataset):
    """Imagen completa con padding a multiplo de 32; las metricas se calculan sin padding."""

    def __init__(self, samples) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = load_image(s.image_path)
        mask = load_mask_binary(s.mask_path)
        pads = pad_amounts(mask.shape[0], mask.shape[1], MULTIPLE)
        img = pad_hw(img, pads)
        mask = pad_hw(mask, pads)
        return {
            "image": torch.from_numpy(img).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask).unsqueeze(0).contiguous(),
            "pads": torch.tensor(pads, dtype=torch.long),
            "sample_id": s.sample_id,
        }


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """0.5 * BCE + 0.5 * (1 - Dice suave), por muestra y promediado en el batch."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()
    return 0.5 * bce + 0.5 * dice


def train_one_epoch(model, loader, optimizer, device, freeze_encoder: bool, max_batches: int | None = None) -> float:
    """Una pasada sobre el loader. Si el encoder esta congelado, su BatchNorm queda en eval.

    max_batches: limite opcional de lotes (solo para pruebas de humo; None = epoca completa).
    """
    model.train()
    if freeze_encoder:
        model.encoder.eval()
    total, n = 0.0, 0
    for batch in loader:
        if max_batches is not None and n >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = dice_bce_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5, max_batches: int | None = None) -> dict:
    model.eval()
    conf = Confusion()
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        loss_sum += dice_bce_loss(logits, y).item()
        n_batches += 1
        pred = (torch.sigmoid(logits) > threshold).cpu().numpy()[:, 0]
        target = batch["mask"].numpy()[:, 0]
        pads = batch["pads"].numpy()
        H, W = pred.shape[1], pred.shape[2]
        for i in range(pred.shape[0]):
            top, bottom, left, right = pads[i]
            conf.update(
                pred[i, top : H - bottom, left : W - right],
                target[i, top : H - bottom, left : W - right],
            )
    scores = conf.scores()
    scores["loss"] = loss_sum / max(n_batches, 1)
    return scores


def make_optimizer(model, phase: str, lr_encoder: float, lr_decoder: float, wd: float):
    if phase == "phase1_frozen":
        return torch.optim.AdamW(decoder_parameters(model), lr=lr_decoder, weight_decay=wd)
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters(model), "lr": lr_encoder},
            {"params": decoder_parameters(model), "lr": lr_decoder},
        ],
        weight_decay=wd,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "unet_resnet34_v1")
    ap.add_argument("--phase1-epochs", type=int, default=4)
    ap.add_argument("--phase2-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--lr-decoder", type=float, default=1e-3)
    ap.add_argument("--lr-encoder", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--evaluate-test", action="store_true")
    # Solo para pruebas de humo: limitan lotes por epoca. None = comportamiento normal.
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, weight_report = build_model(WEIGHTS_PATH)
    model.to(device)

    args.out.mkdir(parents=True, exist_ok=True)
    run_info = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "smp": smp.__version__,
        "device": str(device),
        "weights": weight_report,
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_encoder": sum(p.numel() for p in encoder_parameters(model)),
        "params_decoder": sum(p.numel() for p in decoder_parameters(model)),
        "multiple": MULTIPLE,
        "crop_size_train": CROP_SIZE,
    }
    (args.out / "config.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")

    train_samples = read_split(args.root, "train")
    val_samples = read_split(args.root, "val")
    train_ds = TrainDataset(train_samples, seed=args.seed)
    val_ds = EvalDataset(val_samples)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
    )

    phases = [
        ("phase1_frozen", args.phase1_epochs, True),
        ("phase2_finetune", args.phase2_epochs, False),
    ]

    best_dice = -1.0
    history: list[dict] = []
    global_epoch = 0
    for phase, n_epochs, frozen in phases:
        if n_epochs <= 0:
            continue
        set_encoder_trainable(model, trainable=not frozen)
        optimizer = make_optimizer(model, phase, args.lr_encoder, args.lr_decoder, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        print(f"{phase}: {n_epochs} epocas, parametros entrenables={count_trainable(model)}")

        for ep in range(1, n_epochs + 1):
            global_epoch += 1
            train_ds.epoch = global_epoch
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, freeze_encoder=frozen,
                max_batches=args.max_train_batches,
            )
            val = evaluate(model, val_loader, device, max_batches=args.max_val_batches)
            record = {
                "global_epoch": global_epoch,
                "phase": phase,
                "phase_epoch": ep,
                "lr_groups": [g_["lr"] for g_ in optimizer.param_groups],
                "trainable_params": count_trainable(model),
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_iou": val["iou"],
                "val_dice": val["dice"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "seconds": round(time.time() - t0, 1),
            }
            history.append(record)
            print(json.dumps(record))

            scheduler.step()
            ckpt = {
                "global_epoch": global_epoch,
                "phase": phase,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": run_info,
                "val_dice": val["dice"],
            }
            torch.save(ckpt, args.out / "last.pt")
            if val["dice"] > best_dice:
                best_dice = val["dice"]
                torch.save(ckpt, args.out / "best.pt")
                print(f"  nuevo mejor val_dice={best_dice:.4f} -> best.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"mejor val_dice: {best_dice:.4f}")

    if args.evaluate_test:
        best = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        test_ds = EvalDataset(read_split(args.root, "test"))
        test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        test = evaluate(model, test_loader, device)
        (args.out / "test_results.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
        print("TEST (evaluacion unica):", json.dumps(test))


if __name__ == "__main__":
    main()
