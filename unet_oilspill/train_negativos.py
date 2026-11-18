"""Reentrenamiento corto desde best.pt mezclando teselas originales (con mascara) y negativos CSIRO (mascara vacia).

- Lote de 4 = 2 recortes de teselas originales + 2 recortes de negativos CSIRO (1 a 1).
- Negativos por epoca = n-neg (por defecto 108); GBR a lo sumo 50 % de ellos.
- Augmentation: giros y espejo (misma transformacion en imagen y mascara).
- LR 1e-4 decoder / 1e-5 encoder, coseno, semilla 42. No sobrescribe checkpoints existentes.
- Cada epoca: recalibra BatchNorm del decoder (igual que la plataforma, con imagenes de entrenamiento),
  mide Dice/precision en val original y falsas alarmas en val CSIRO (negativos). Seleccion: menos falsas
  alarmas con Dice >= base - 0,03. NO toca el test original ni la region SIN.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import data as D
import fa_eval as F
from app import inference
from model import decoder_parameters, encoder_parameters, set_encoder_trainable
from train import dice_bce_loss, set_seed


def augment(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator):
    k = int(rng.integers(0, 4))
    img, msk = np.rot90(img, k), np.rot90(msk, k)
    if rng.integers(0, 2):
        img, msk = img[:, ::-1], msk[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


class MixDataset(Dataset):
    def __init__(self, orig, neg_rows, seed):
        self.orig, self.neg_rows, self.seed, self.epoch = orig, neg_rows, seed, 0
        self.items: list[tuple[str, int]] = []
        self.n_gbr = self.n_other = 0

    def set_epoch(self, epoch: int, n_neg: int, gbr_max_frac: float = 0.5):
        rng = np.random.default_rng([self.seed, epoch])
        gbr = [i for i, r in enumerate(self.neg_rows) if r["region"] == "GBR"]
        oth = [i for i, r in enumerate(self.neg_rows) if r["region"] != "GBR"]
        n_gbr = min(len(gbr), int(n_neg * gbr_max_frac))
        n_oth = min(len(oth), n_neg - n_gbr)
        # GBR nunca supera el 50 % de los negativos usados: si faltan de otras regiones, se reduce el total.
        n_gbr = min(len(gbr), n_neg - n_oth, n_oth)
        neg = [int(i) for i in rng.choice(gbr, n_gbr, replace=False)] + \
              [int(i) for i in rng.choice(oth, n_oth, replace=False)]
        rng.shuffle(neg)
        pos = [int(i) for i in rng.permutation(len(self.orig))]
        pos = (pos * (len(neg) // len(pos) + 1))[: len(neg)]
        self.items = []
        for i in range(0, len(neg) - 1, 2):  # [pos, pos, neg, neg]
            self.items += [("p", pos[i]), ("p", pos[i + 1]), ("n", neg[i]), ("n", neg[i + 1])]
        self.epoch = epoch
        self.n_gbr, self.n_other = n_gbr, n_oth

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kind, j = self.items[idx]
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        if kind == "p":
            s = self.orig[j]
            img, mask = D.load_image(s.image_path), D.load_mask_binary(s.mask_path)
        else:
            rgb = F.load_csiro_rgb(F.CSIRO_ROOT / self.neg_rows[j]["path"]).astype(np.float32) / 255.0
            img = (rgb - D.IMAGENET_MEAN) / D.IMAGENET_STD
            mask = np.zeros(img.shape[:2], dtype=np.float32)
        top, left = D.random_crop_coords(mask.shape[0], mask.shape[1], D.CROP_SIZE, rng)
        img, mask = D.crop_pair(img, mask, top, left, D.CROP_SIZE)
        img, mask = augment(img, mask, rng)
        return {"image": torch.from_numpy(img).permute(2, 0, 1).contiguous(),
                "mask": torch.from_numpy(mask).unsqueeze(0).contiguous()}


def recalibrate_bn(model, train_samples) -> dict:
    """Mismo procedimiento que platform/tools/recalibrate_bn.py (imagenes de entrenamiento, sin etiquetas)."""
    bns = [m for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d) and not n.startswith("encoder.")]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
    model.eval()
    for b in bns:
        b.train()
    with torch.no_grad():
        for s in train_samples:
            rgb = np.asarray(Image.open(s.image_path).convert("RGB")).astype(np.float32) / 255.0
            arr = (rgb - D.IMAGENET_MEAN) / D.IMAGENET_STD
            pads = D.pad_amounts(*arr.shape[:2], D.MULTIPLE)
            x = torch.from_numpy(np.ascontiguousarray(D.pad_hw(arr, pads))).permute(2, 0, 1).unsqueeze(0)
            model(x)
    model.eval()
    return {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("encoder.")
            and ("running_mean" in k or "running_var" in k or "num_batches_tracked" in k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("runs/unet_resnet34_demo_12ep/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/unet_resnet34_negativos"))
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--n-neg", type=int, default=108)
    ap.add_argument("--val-neg-limit", type=int, default=0, help="0 = todos los negativos de validacion")
    ap.add_argument("--base-dice", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--time-cap-min", type=float, default=0.0, help="0 = sin tope")
    args = ap.parse_args()
    if (args.out / "best.pt").exists():
        raise SystemExit("best.pt ya existe: no se sobrescribe")
    set_seed(args.seed)

    seg = inference.Segmenter(checkpoint=args.base, bn_recal=None)  # best.pt con pesos crudos
    model = seg._model
    train_orig = D.read_split(D.DEFAULT_DATASET_ROOT, "train")
    neg_train = F.read_index("train", 0)
    neg_val = F.read_index("val", 0)
    if args.val_neg_limit:
        rng = np.random.default_rng(0)
        neg_val = [neg_val[i] for i in sorted(rng.choice(len(neg_val), args.val_neg_limit, replace=False))]
    ds = MixDataset(train_orig, neg_train, args.seed)
    if args.freeze_encoder:
        groups = [{"params": decoder_parameters(model), "lr": 1e-4}]
    else:
        groups = [{"params": encoder_parameters(model), "lr": 1e-5},
                  {"params": decoder_parameters(model), "lr": 1e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    set_encoder_trainable(model, not args.freeze_encoder)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "batch": "2 originales + 2 negativos",
         "lr_encoder": 1e-5, "lr_decoder": 1e-4, "gbr_max_frac_of_negatives": 0.5,
         "val_neg_n": len(neg_val), "train_neg_pool": len(neg_train)}, indent=2), encoding="utf-8")

    history, best = [], None
    run_t0 = time.time()
    cap = args.time_cap_min * 60
    train_budget = max(30.0, cap / args.epochs - 150) if cap else None
    floor = args.base_dice - 0.03
    for ep in range(1, args.epochs + 1):
        if cap and time.time() - run_t0 > cap:
            print(f"parada anticipada: tope de {args.time_cap_min} min superado antes de la epoca {ep}", flush=True)
            break
        t0 = time.time()
        ds.set_epoch(ep, args.n_neg)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        tot, n = 0.0, 0
        for b in loader:
            if train_budget and time.time() - t0 > train_budget:
                break
            opt.zero_grad(set_to_none=True)
            loss = dice_bce_loss(model(b["image"]), b["mask"])
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        sched.step()
        t_train = time.time() - t0
        raw_state = copy.deepcopy(model.state_dict())
        torch.save({"global_epoch": ep, "phase": "negativos_csiro", "model": raw_state}, args.out / "last.pt")
        bn_state = recalibrate_bn(model, train_orig)
        val = F.eval_val_original(seg)
        fa = F.rate(F.eval_csiro(seg, neg_val))
        rec = {"epoch": ep, "train_loss": tot / max(n, 1), "neg_gbr": ds.n_gbr, "neg_otras": ds.n_other,
               "val_dice": val["dice"], "val_precision": val["precision"], "val_recall": val["recall"],
               "fa_val": fa["rate"], "fa_val_n": fa["n"], "sec_train": round(t_train), "batches": n,
               "sec_total": round(time.time() - t0)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        ok = val["dice"] >= floor
        if best is None:
            better = True
        elif ok and not best["ok"]:
            better = True
        elif ok and best["ok"]:
            better = (fa["rate"], -val["dice"]) < (best["fa"], -best["dice"])
        elif not ok and not best["ok"]:
            better = val["dice"] > best["dice"]
        else:
            better = False
        if better:
            best = {"epoch": ep, "ok": ok, "fa": fa["rate"], "dice": val["dice"]}
            torch.save({"global_epoch": ep, "phase": "negativos_csiro", "model": raw_state,
                        "val_dice": val["dice"], "fa_val": fa["rate"]}, args.out / "best.pt")
            torch.save({"state": bn_state, "n_images": len(train_orig), "split": "train"},
                       args.out / "bn_decoder_recal.pt")
            print(f"  -> mejor (epoca {ep}, cumple Dice: {ok})", flush=True)
        model.load_state_dict(raw_state)  # seguir entrenando con BN sin recalibrar
    (args.out / "history.json").write_text(
        json.dumps({"history": history, "selected": best, "dice_floor": floor}, indent=2), encoding="utf-8")
    print("seleccionado:", best)


if __name__ == "__main__":
    main()
