"""Entrenamiento final: imagenes CON derrame (173 originales + 1.000 SOS Sentinel-1) y SIN derrame (1.500 CSIRO clase 0).

Etapas (python train_final.py <etapa>):
  split    -> arma el reparto por GRUPOS (nunca imagen por imagen) y lo guarda en runs/unet_resnet34_final/split.json
  train    -> parte de unet_resnet34_demo_12ep/best.pt; entrena con train, elige el mejor con validacion
  compare  -> mide modelo actual vs nuevo en la MISMA prueba (una sola vez)

Criterio de grupos (cada grupo va entero a un solo conjunto):
  originales: escena de origen = prefijo IMG_XX del nombre (las teselas IMG_12_TILE_001.. salen de la misma imagen)
  SOS: no trae escena; se agrupa por bloques contiguos de ids (SOS_BLOCK) y se descartan SOS_MARGIN ids a cada lado de
       cada frontera entre bloques de conjuntos distintos
  CSIRO: codigo de region al final del nombre (GBR, SIN, JAV...); cada region va a un solo conjunto
"""
from __future__ import annotations

import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "platform"))
sys.dont_write_bytecode = True

import data as D  # noqa: E402
from app import analysis as A  # noqa: E402
from app import config, inference  # noqa: E402
from model import decoder_parameters, encoder_parameters, set_encoder_trainable  # noqa: E402
from train import dice_bce_loss, set_seed  # noqa: E402

# ---------------- parametros faciles de cambiar ----------------
PCT = (0.70, 0.15, 0.15)   # entrenamiento / validacion / prueba (dentro de cada grupo A y B)
SEED = 42
N_SPILL_SOS = 1000         # imagenes SOS con mancha
N_CLEAN = 1500             # imagenes CSIRO sin derrame
SOS_BLOCK, SOS_MARGIN = 100, 30
BUDGET_MIN = 30.0
NAMES = ("train", "val", "test")

SOS = Path(r"C:\Users\ANDAMIOS\Desktop\SOS_extract")  # fuera de los repos (solo Sentinel-1)
CSIRO = Path(r"C:\Users\ANDAMIOS\Desktop\OCEAN-GUARD-AI\datasets_extra\csiro")  # solo lectura
OUT = HERE / "runs" / "unet_resnet34_final"
BASE_CKPT = HERE / "runs" / "unet_resnet34_demo_12ep" / "best.pt"
BASE_BN = ROOT / "platform" / "assets" / "bn_decoder_recal.pt"


# ---------------- lectura ----------------
def sos_paths(sid: int):
    if (SOS / "train" / "sentinel" / f"{sid}_sat.jpg").exists():
        return SOS / "train" / "sentinel" / f"{sid}_sat.jpg", SOS / "train" / "sentinel" / f"{sid}_mask.png"
    return SOS / "test" / "sentinel" / "sat" / f"{sid}_sat.jpg", SOS / "test" / "sentinel" / "gt" / f"{sid}_mask.png"


def gray_rgb(path: Path) -> np.ndarray:
    """SAR en gris, 3 canales iguales, uint8 HxWx3 (como las imagenes del proyecto)."""
    a = np.asarray(Image.open(path).convert("RGB"))
    if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
        g = np.asarray(Image.open(path).convert("L"))
        a = np.stack([g, g, g], axis=-1)
    return a


def binmask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


# ---------------- reparto por grupos ----------------
def assign_groups(sizes: dict, pct=PCT, seed=SEED, min_groups=2, iters=40000) -> dict:
    """Asigna cada grupo entero a un conjunto, acercandose a pct en cantidad de imagenes (busqueda aleatoria con semilla)."""
    keys = sorted(sizes)
    total = sum(sizes.values())
    target = [p * total for p in pct]
    rng = np.random.default_rng(seed)
    best, best_cost = None, 1e18
    for _ in range(iters):
        a = rng.choice(len(pct), size=len(keys), p=pct)
        if any((a == s).sum() < min_groups for s in range(len(pct))):
            continue
        cnt = [sum(sizes[k] for k, s in zip(keys, a) if s == i) for i in range(len(pct))]
        cost = sum(abs(c - t) for c, t in zip(cnt, target))
        if cost < best_cost:
            best, best_cost = a.copy(), cost
    return {k: int(s) for k, s in zip(keys, best)}


def stage_split() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    # ---- A1: originales, por escena IMG_XX ----
    orig = [s for sp in ("train", "val", "test") for s in D.read_split(D.DEFAULT_DATASET_ROOT, sp)]
    scenes: dict[str, list[str]] = {}
    for s in orig:
        scenes.setdefault(s.sample_id.split("_TILE")[0], []).append(s.sample_id)
    a_orig = assign_groups({k: len(v) for k, v in scenes.items()})
    orig_split = {n: sorted(i for k, v in scenes.items() if a_orig[k] == si for i in v) for si, n in enumerate(NAMES)}
    # ---- A2: SOS Sentinel-1, bloques contiguos + margen, 1.000 con mancha al azar ----
    ids = sorted(int(p.name.split("_")[0]) for base in (SOS / "train" / "sentinel", SOS / "test" / "sentinel" / "sat") for p in base.glob("*_sat.jpg"))
    nonempty = {i: bool(binmask(sos_paths(i)[1]).any()) for i in ids}
    blocks: dict[int, list[int]] = {}
    for pos, i in enumerate(ids):
        blocks.setdefault(pos // SOS_BLOCK, []).append(pos)
    a_blk = assign_groups({b: sum(nonempty[ids[p]] for p in ps) for b, ps in blocks.items()}, min_groups=3)
    pool = {n: [] for n in NAMES}
    dropped = 0
    for pos, i in enumerate(ids):
        if not nonempty[i]:
            continue
        b, off = divmod(pos, SOS_BLOCK)
        s = a_blk[b]
        edge_lo = off < SOS_MARGIN and (b - 1) in a_blk and a_blk[b - 1] != s
        edge_hi = (SOS_BLOCK - off) <= SOS_MARGIN and (b + 1) in a_blk and a_blk[b + 1] != s
        if edge_lo or edge_hi:
            dropped += 1
            continue
        pool[NAMES[s]].append(i)
    want = [round(N_SPILL_SOS * p) for p in PCT]
    want[0] = N_SPILL_SOS - want[1] - want[2]
    sos_split = {n: sorted(rng.sample(pool[n], min(w, len(pool[n])))) for n, w in zip(NAMES, want)}
    sos_blocks = {n: sorted({pos // SOS_BLOCK for pos, i in enumerate(ids) if i in set(sos_split[n])}) for n in NAMES}
    # ---- B: CSIRO clase 0, reparto por region ----
    rows = [r for r in csv.DictReader((CSIRO / "index.csv").open(encoding="utf-8")) if r["label"] == "0"]
    by_reg: dict[str, list[str]] = {}
    for r in rows:
        by_reg.setdefault(r["region"], []).append(r["path"])
    cap = 1
    while sum(min(len(v), cap) for v in by_reg.values()) < N_CLEAN:
        cap += 1
    take = {k: min(len(v), cap) for k, v in by_reg.items()}
    over = sum(take.values()) - N_CLEAN
    for k in sorted(take, key=lambda k: -take[k]):  # recorta de las regiones topeadas hasta llegar a N_CLEAN exacto
        if over <= 0:
            break
        if take[k] == cap:
            take[k] -= 1
            over -= 1
    chosen = {k: sorted(rng.sample(sorted(v), take[k])) for k, v in by_reg.items()}
    a_reg = assign_groups(take, min_groups=3)
    csiro_split = {n: sorted(p for k, v in chosen.items() if a_reg[k] == si for p in v) for si, n in enumerate(NAMES)}
    regions = {n: sorted(k for k in take if a_reg[k] == si) for si, n in enumerate(NAMES)}
    # ---- verificacion: ningun grupo en dos conjuntos ----
    for groupname, per in (("escenas originales", {n: sorted({i.split('_TILE')[0] for i in orig_split[n]}) for n in NAMES}),
                           ("bloques SOS", sos_blocks), ("regiones CSIRO", regions)):
        for x in range(3):
            for y in range(x + 1, 3):
                assert not (set(per[NAMES[x]]) & set(per[NAMES[y]])), f"{groupname} repetido en {NAMES[x]} y {NAMES[y]}"
    # margen SOS: distancia minima entre ids de conjuntos distintos (en posiciones de la lista ordenada)
    posmap = {i: p for p, i in enumerate(ids)}
    min_gap = min(abs(posmap[a] - posmap[b]) for x in range(3) for y in range(x + 1, 3)
                  for a in sos_split[NAMES[x]] for b in sos_split[NAMES[y]])
    assert min_gap > SOS_MARGIN, min_gap
    counts = {n: {"con_derrame": len(orig_split[n]) + len(sos_split[n]), "originales": len(orig_split[n]), "sos": len(sos_split[n]),
                  "sin_derrame": len(csiro_split[n])} for n in NAMES}
    out = {"params": {"pct": PCT, "seed": SEED, "n_spill_sos": N_SPILL_SOS, "n_clean": N_CLEAN, "sos_block": SOS_BLOCK,
                      "sos_margin": SOS_MARGIN},
           "criterio": {"originales": "escena de origen = prefijo IMG_XX", "sos": f"bloques contiguos de {SOS_BLOCK} ids con margen de {SOS_MARGIN}",
                        "csiro": "region (codigo al final del nombre)"},
           "counts": counts, "verificacion": {"grupos_en_dos_conjuntos": 0, "sos_distancia_minima_entre_conjuntos_en_ids": min_gap,
                                             "sos_descartados_por_margen": dropped},
           "grupos": {"escenas_originales": {n: sorted({i.split('_TILE')[0] for i in orig_split[n]}) for n in NAMES},
                      "regiones_csiro": regions, "sos_bloques": sos_blocks, "csiro_por_region_elegidas": take},
           "items": {n: {"originales": orig_split[n], "sos": sos_split[n], "csiro": csiro_split[n]} for n in NAMES}}
    (OUT / "split.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for n in NAMES:
        c = counts[n]
        print(f"{n}: con derrame {c['con_derrame']} (originales {c['originales']} + SOS {c['sos']}) | sin derrame {c['sin_derrame']} | total {c['con_derrame'] + c['sin_derrame']}")
    print("escenas originales:", out["grupos"]["escenas_originales"])
    print("regiones CSIRO:", regions)
    print("verificacion OK: ningun grupo en dos conjuntos; distancia minima SOS entre conjuntos =", min_gap, "ids; descartados por margen:", dropped)


# ---------------- evaluacion (regla de la plataforma) ----------------
def final_mask(seg, rgb):
    prob = seg.predict_prob(rgb)
    return A.remove_small_components(prob > config.DEFAULT_THRESHOLD, config.MIN_COMPONENT_PX)


def load_eval_items(names, split_items):
    """Lista de (rgb, mascara, fuente) para los conjuntos pedidos."""
    items = []
    by_id = {s.sample_id: s for sp in ("train", "val", "test") for s in D.read_split(D.DEFAULT_DATASET_ROOT, sp)}
    for n in names:
        for sid in split_items[n]["originales"]:
            s = by_id[sid]
            items.append((np.asarray(Image.open(s.image_path).convert("RGB")), D.load_mask_binary(s.mask_path) > 0, "originales"))
        for i in split_items[n]["sos"]:
            ip, mp = sos_paths(i)
            items.append((gray_rgb(ip), binmask(mp), "sos"))
        for p in split_items[n]["csiro"]:
            a = np.asarray(Image.open(CSIRO / p).convert("RGB"))
            items.append((np.stack([a[..., 0]] * 3, axis=-1), np.zeros(a.shape[:2], dtype=bool), "csiro"))
    return items


def _acc():
    return {"tp": 0, "fp": 0, "fn": 0, "img": 0, "marked": 0}


def _fin(a):
    d = {"imagenes": a["img"], "dice": 2 * a["tp"] / max(2 * a["tp"] + a["fp"] + a["fn"], 1),
         "precision": a["tp"] / max(a["tp"] + a["fp"], 1), "recall": a["tp"] / max(a["tp"] + a["fn"], 1),
         "imagenes_marcadas": a["marked"], "tasa_marcadas": a["marked"] / a["img"] if a["img"] else None}
    return d


def eval_items(seg, items) -> dict:
    """con_derrame: Dice/precision/recall + deteccion; sin_derrame: falsa alarma; conjunto: Dice/precision/recall con todo."""
    spill, clean, full = _acc(), _acc(), _acc()
    by_src = {}
    for rgb, mask, src in items:
        m = final_mask(seg, rgb)
        marked = bool(m.any())
        tp, fp, fn = int((m & mask).sum()), int((m & ~mask).sum()), int((~m & mask).sum())
        for acc in (spill if mask.any() else clean, full):
            acc["tp"] += tp; acc["fp"] += fp; acc["fn"] += fn; acc["img"] += 1; acc["marked"] += int(marked)
        if mask.any():
            s = by_src.setdefault(src, _acc())
            s["tp"] += tp; s["fp"] += fp; s["fn"] += fn; s["img"] += 1; s["marked"] += int(marked)
    out = {"con_derrame": _fin(spill), "sin_derrame": _fin(clean), "conjunto_completo": _fin(full),
           "con_derrame_por_fuente": {k: _fin(v) for k, v in by_src.items()}}
    out["con_derrame"]["deteccion"] = out["con_derrame"].pop("tasa_marcadas")
    out["sin_derrame"]["falsa_alarma"] = out["sin_derrame"].pop("tasa_marcadas")
    for k in ("dice", "precision", "recall"):
        out["sin_derrame"].pop(k)
    return out


def recalibrate_bn(model, samples) -> dict:
    """Igual que platform/tools/recalibrate_bn.py: estadisticas BN del decoder con imagenes de entrenamiento, sin etiquetas."""
    bns = [m for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d) and not n.startswith("encoder.")]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
    model.eval()
    for b in bns:
        b.train()
    with torch.no_grad():
        for s in samples:
            rgb = np.asarray(Image.open(s.image_path).convert("RGB")).astype(np.float32) / 255.0
            arr = (rgb - D.IMAGENET_MEAN) / D.IMAGENET_STD
            pads = D.pad_amounts(*arr.shape[:2], D.MULTIPLE)
            model(torch.from_numpy(np.ascontiguousarray(D.pad_hw(arr, pads))).permute(2, 0, 1).unsqueeze(0))
    model.eval()
    return {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("encoder.")
            and ("running_mean" in k or "running_var" in k or "num_batches_tracked" in k)}


# ---------------- entrenamiento ----------------
def augment(img, msk, rng):
    k = int(rng.integers(0, 4))
    img, msk = np.rot90(img, k), np.rot90(msk, k)
    if rng.integers(0, 2):
        img, msk = img[:, ::-1], msk[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


class TrainSet(Dataset):
    def __init__(self, items, seed):
        by_id = {s.sample_id: s for sp in ("train", "val", "test") for s in D.read_split(D.DEFAULT_DATASET_ROOT, sp)}
        self.items = [("o", by_id[i]) for i in items["originales"]] + [("s", i) for i in items["sos"]] + [("c", p) for p in items["csiro"]]
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kind, ref = self.items[idx]
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        if kind == "o":
            img, mask = D.load_image(ref.image_path), D.load_mask_binary(ref.mask_path)
        elif kind == "s":
            ip, mp = sos_paths(ref)
            img = (gray_rgb(ip).astype(np.float32) / 255.0 - D.IMAGENET_MEAN) / D.IMAGENET_STD
            mask = binmask(mp).astype(np.float32)
        else:
            a = np.asarray(Image.open(CSIRO / ref).convert("RGB"))
            img = (np.stack([a[..., 0]] * 3, axis=-1).astype(np.float32) / 255.0 - D.IMAGENET_MEAN) / D.IMAGENET_STD
            mask = np.zeros(a.shape[:2], dtype=np.float32)
        if mask.shape[0] > D.CROP_SIZE or mask.shape[1] > D.CROP_SIZE:
            top, left = D.random_crop_coords(mask.shape[0], mask.shape[1], D.CROP_SIZE, rng)
            img, mask = D.crop_pair(img, mask, top, left, D.CROP_SIZE)
        img, mask = augment(img, mask, rng)
        return {"image": torch.from_numpy(img).permute(2, 0, 1).contiguous(), "mask": torch.from_numpy(mask).unsqueeze(0).contiguous()}


def stage_train() -> None:
    set_seed(SEED)
    sp = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
    ds = TrainSet(sp["items"]["train"], SEED)
    val_items = load_eval_items(["val"], sp["items"])
    seg = inference.Segmenter(checkpoint=BASE_CKPT, bn_recal=None)
    model = seg._model
    set_encoder_trainable(model, True)
    print(f"train: {len(ds)} imagenes | validacion: {len(val_items)} | device: cpu", flush=True)
    tmp = copy.deepcopy(model)
    topt = torch.optim.AdamW([{"params": encoder_parameters(tmp), "lr": 1e-5}, {"params": decoder_parameters(tmp), "lr": 1e-4}])
    tmp.train()
    t0, n_t = time.time(), 0
    for b in DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0)):
        topt.zero_grad()
        dice_bce_loss(tmp(b["image"]), b["mask"]).backward()
        topt.step()
        n_t += 1
        if n_t == 12:
            break
    per_batch = (time.time() - t0) / n_t
    del tmp, topt
    t_train = per_batch * ((len(ds) + 3) // 4)
    t_val = 0.2 * len(val_items)
    epochs = int(max(1, min(6, (BUDGET_MIN * 60) // (t_train + t_val))))
    print(f"velocidad: {per_batch:.2f} s/lote -> epoca ~{t_train/60:.1f} min + validacion ~{t_val/60:.1f} min; epocas = {epochs}", flush=True)
    opt = torch.optim.AdamW([{"params": encoder_parameters(model), "lr": 1e-5}, {"params": decoder_parameters(model), "lr": 1e-4}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, history = None, []
    t_run = time.time()
    for ep in range(1, epochs + 1):
        ds.epoch = ep
        t0 = time.time()
        model.train()
        tot = n = 0
        for b in DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(SEED + ep)):
            opt.zero_grad(set_to_none=True)
            loss = dice_bce_loss(model(b["image"]), b["mask"])
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        sched.step()
        model.eval()
        val = eval_items(seg, val_items)
        score = val["conjunto_completo"]["dice"] - 0.5 * val["sin_derrame"]["falsa_alarma"]  # criterio de seleccion
        rec = {"epoch": ep, "train_loss": tot / n, "score_seleccion": score, "val": val, "seconds": round(time.time() - t0)}
        history.append(rec)
        print(json.dumps({"epoch": ep, "loss": round(tot / n, 4), "score": round(score, 4),
                          "dice_conjunto": round(val["conjunto_completo"]["dice"], 4), "falsa_alarma": round(val["sin_derrame"]["falsa_alarma"], 4),
                          "deteccion": round(val["con_derrame"]["deteccion"], 4), "seg": rec["seconds"]}), flush=True)
        if best is None or score > best["score"]:
            best = {"epoch": ep, "score": score}
            torch.save({"global_epoch": ep, "phase": "final", "model": copy.deepcopy(model.state_dict()), "score": score}, OUT / "best.pt")
            print(f"  -> mejor (epoca {ep})", flush=True)
        if time.time() - t_run > BUDGET_MIN * 60 * 1.15 and ep < epochs:
            print("parada anticipada por tiempo", flush=True)
            break
    ck = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    by_id = {s.sample_id: s for sp_ in ("train", "val", "test") for s in D.read_split(D.DEFAULT_DATASET_ROOT, sp_)}
    bn_imgs = [by_id[i] for i in sp["items"]["train"]["originales"]]  # solo originales de ENTRENAMIENTO nuevo
    torch.save({"state": recalibrate_bn(model, bn_imgs), "n_images": len(bn_imgs), "split": "train(final)"}, OUT / "bn_decoder_recal.pt")
    (OUT / "history.json").write_text(json.dumps({"history": history, "seleccionado": best, "epocas": epochs, "segundos_por_lote": per_batch}, indent=1), encoding="utf-8")
    print("listo; seleccionado:", best, flush=True)


def stage_compare(nuevo: Path | None = None, bn: Path | None = None, out: Path | None = None) -> None:
    nuevo, bn, out = nuevo or OUT / "best.pt", bn or OUT / "bn_decoder_recal.pt", out or OUT / "comparacion.json"
    sp = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
    items = load_eval_items(["test"], sp["items"])
    seen = set(s.sample_id for s in D.read_split(D.DEFAULT_DATASET_ROOT, "train"))  # originales con los que se entreno el modelo actual
    res = {"regla": {"umbral": config.DEFAULT_THRESHOLD, "zona_minima_px": config.MIN_COMPONENT_PX},
           "prueba": sp["counts"]["test"],
           "aviso": ("el modelo ACTUAL se entreno con las 108 teselas originales del split train; las originales de esta prueba "
                     "que estan en ese split le dan ventaja"),
           "originales_en_prueba_vistas_por_el_actual": sum(1 for i in sp["items"]["test"]["originales"] if i in seen),
           "modelos": {}}
    for name, ck, bn in (("actual", BASE_CKPT, BASE_BN), ("nuevo_final", nuevo, bn)):
        seg = inference.Segmenter(checkpoint=ck, bn_recal=bn)
        assert seg.info["bn_decoder_recalibrated"]
        res["modelos"][name] = eval_items(seg, items)  # prueba: UNA sola vez por modelo
        print(name, json.dumps({k: {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items() if not isinstance(x, dict)}
                                for k, v in res["modelos"][name].items() if k != "con_derrame_por_fuente"}), flush=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("comparacion guardada en", out, flush=True)


if __name__ == "__main__":
    {"split": stage_split, "train": stage_train, "compare": stage_compare}[sys.argv[1]]()
