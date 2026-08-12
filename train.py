"""
Train one (architecture, seed) pair.

    python train.py --arch vit --seed 42
    python train.py --arch vss --seed 2024 --data-root /content/cifar_data
    python train.py --arch kan --seed 42 --smoke      # 2 epochs on fake data

Nine runs total: {vit, kan, vss} x {42, 123, 2024}. Every run writes
checkpoints/{arch}_seed{seed}.pt and SKIPS if that file already exists, so
re-running a cell after a Colab disconnect resumes rather than restarts.

WHAT IS HELD IDENTICAL ACROSS ALL NINE RUNS
Optimizer, LR, schedule, epochs, batch size, augmentation, label smoothing,
gradient clipping, precision, and -- within a seed -- the data order and the
shared stem initialisation. The ONLY thing that varies is the block. That is
what licenses the claim that measured representational differences are
attributable to the architectural mechanism.

WHY NO AMP
selective_scan_torch hard-casts to fp32 internally and KAN's b_splines needs
fp32, so autocast would accelerate ONLY ViT -- a precision asymmetry in an
experiment whose validity rests on the three models being identical apart
from the block. fp32 everywhere costs ViT some wall-clock and removes the
confound.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from make_splits import load_splits
from models.stem import MotivationNet, count_params
from models.block_vit import make_vit_block
from models.block_kan import make_kan_block
from models.block_vss import make_vss_block

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG that can affect the run.

    Deliberately NOT calling torch.use_deterministic_algorithms(True): it
    throws on the scatter/index ops inside the selective scan and buys
    nothing here. Bitwise reproducibility is not achievable on GPU anyway
    (see the lstsq note in COLAB.md); a fixed, reported seed set is the
    standard we hold to, and every number is reported as mean +/- std over
    three seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_loaders(seed: int, data_root: str, smoke: bool = False):
    """CIFAR-100 with the frozen 45k/5k split.

    Two dataset instances are needed because train and val use different
    transforms; both are Subset-ed with the SAME frozen index arrays.
    """
    if smoke:
        # Deliberately imports nothing beyond torch, so --smoke validates the
        # training loop on any machine, including one without torchvision.
        from torch.utils.data import TensorDataset
        n = 512
        g = torch.Generator().manual_seed(0)
        xs = torch.randn(n, 3, 32, 32, generator=g)
        ys = torch.randint(0, config.NUM_CLASSES, (n,), generator=g)
        ds = TensorDataset(xs, ys)
        tr_ds, va_ds = Subset(ds, range(384)), Subset(ds, range(384, n))
    else:
        import torchvision.transforms as T
        from torchvision.datasets import CIFAR100

        norm = T.Normalize(config.CIFAR100_MEAN, config.CIFAR100_STD)
        train_tf = T.Compose([
            T.RandomCrop(config.IMG_SIZE,
                         padding=config.AUGMENT["random_crop_padding"]),
            T.RandomHorizontalFlip(),
            T.ToTensor(), norm,
        ])
        eval_tf = T.Compose([T.ToTensor(), norm])

        tr_full = CIFAR100(data_root, train=True, download=True, transform=train_tf)
        va_full = CIFAR100(data_root, train=True, download=True, transform=eval_tf)
        train_idx, val_idx, _ = load_splits(verify=True)
        tr_ds, va_ds = Subset(tr_full, train_idx), Subset(va_full, val_idx)

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=config.TRAIN["num_workers"], pin_memory=True,
                  worker_init_fn=worker_init_fn, persistent_workers=False)
    train_loader = DataLoader(tr_ds, batch_size=config.TRAIN["batch_size"],
                              shuffle=True, drop_last=True, generator=g, **common)
    val_loader = DataLoader(va_ds, batch_size=256, shuffle=False, **common)
    return train_loader, val_loader


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(arch: str, seed: int) -> MotivationNet:
    if arch == "vit":
        cfg = dict(embed_dim=config.EMBED_DIM, **config.VIT)
        fn = make_vit_block(cfg)
    elif arch == "kan":
        cfg = dict(embed_dim=config.EMBED_DIM, n_tokens=config.N_TOKENS, **config.KAN)
        cfg.pop("update_grid")
        fn = make_kan_block(cfg)
    elif arch == "vss":
        cfg = dict(embed_dim=config.EMBED_DIM, grid=config.GRID, **config.VSS)
        cfg.pop("channel_first")
        fn = make_vss_block(cfg)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return MotivationNet(fn, depth=config.DEPTH, embed_dim=config.EMBED_DIM,
                         num_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE,
                         patch_size=config.PATCH_SIZE, in_chans=config.IN_CHANS,
                         stem_seed=seed)


def lr_at(step: int, steps_per_epoch: int) -> float:
    """Linear warmup then cosine, computed per step for a smooth schedule."""
    tr = config.TRAIN
    warm = tr["warmup_epochs"] * steps_per_epoch
    total = tr["epochs"] * steps_per_epoch
    if step < warm:
        return tr["lr"] * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return tr["min_lr"] + 0.5 * (tr["lr"] - tr["min_lr"]) * (1 + np.cos(np.pi * p))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float, float]:
    model.eval()
    loss_sum = n = c1 = c5 = 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += crit(out, y).item()
        top5 = out.topk(5, dim=1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    return loss_sum / n, c1 / n, c5 / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["vit", "kan", "vss"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--data-root", default=os.path.join(ROOT, "cifar_data"))
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on fake data; validates the loop without "
                         "downloading CIFAR-100")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    tag = f"{args.arch}_seed{args.seed}"
    best_path = os.path.join(args.ckpt_dir, f"{tag}.pt")
    last_path = os.path.join(args.ckpt_dir, f"{tag}_last.pt")
    log_path = os.path.join(args.ckpt_dir, f"{tag}_log.json")

    if os.path.exists(best_path) and not args.smoke:
        print(f"[skip] {best_path} exists. Delete it to force a re-run.")
        return

    epochs = 2 if args.smoke else config.TRAIN["epochs"]
    set_seed(args.seed)
    train_loader, val_loader = build_loaders(args.seed, args.data_root, args.smoke)

    set_seed(args.seed)
    model = build_model(args.arch, args.seed).to(device)
    n_params = count_params(model)

    # no weight decay on norms, biases, or the SSM's A_log/D (upstream marks
    # these _no_weight_decay; decaying them would fight the S4D parameterisation)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim <= 1 or getattr(p, "_no_weight_decay", False) or "pos_embed" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.TRAIN["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.TRAIN["lr"])
    crit = nn.CrossEntropyLoss(label_smoothing=config.TRAIN["label_smoothing"])

    start_epoch, best_acc, history = 0, 0.0, []
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best_acc, history = ck["epoch"] + 1, ck["best_acc"], ck["history"]
        print(f"[resume] from epoch {start_epoch} (best so far {best_acc:.4f})")

    spe = len(train_loader)
    print(f"[{tag}] params={n_params:,d}  device={device}  "
          f"epochs={epochs}  steps/epoch={spe}  batch={config.TRAIN['batch_size']}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            lr = lr_at(epoch * spe + i, spe)
            for pg in opt.param_groups:
                pg["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN["grad_clip"])
            opt.step()
            run_loss += loss.item() * y.numel()
            seen += y.numel()

        vl, v1, v5 = evaluate(model, val_loader, device)
        dt = time.time() - t0
        history.append(dict(epoch=epoch, train_loss=run_loss / seen, val_loss=vl,
                            val_top1=v1, val_top5=v5, lr=lr, seconds=dt))

        # KAN grid coverage after the first epoch: activation statistics shift
        # during training, and if coverage falls the spline branch goes
        # inactive and KANLinear degrades toward SiLU+Linear -- i.e. an MLP.
        if args.arch == "kan" and epoch == 0:
            xb = next(iter(val_loader))[0][:64].to(device)
            with torch.no_grad():
                h = model.patch_embed(xb) + model.pos_embed
                covs = []
                for blk in model.blocks:
                    covs.append(min(blk.grid_coverage(h).values()))
                    h = blk(h)
            print(f"    KAN grid coverage after epoch 1: "
                  f"{['%.3f' % c for c in covs]}"
                  + ("   <-- BELOW 0.90, widen grid_range" if min(covs) < 0.90 else ""))
            history[-1]["kan_grid_coverage"] = covs

        if v1 > best_acc:
            best_acc = v1
            torch.save(dict(model=model.state_dict(), arch=args.arch,
                            seed=args.seed, epoch=epoch, val_top1=v1,
                            val_top5=v5, n_params=n_params,
                            config=dict(train=config.TRAIN, depth=config.DEPTH,
                                        embed_dim=config.EMBED_DIM)),
                       best_path)

        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        epoch=epoch, best_acc=best_acc, history=history), last_path)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  ep {epoch:3d}/{epochs}  loss {run_loss/seen:.4f}  "
                  f"val {v1:.4f}/{v5:.4f}  lr {lr:.2e}  {dt:.0f}s")

    print(f"[{tag}] done. best val top-1 = {best_acc:.4f} -> {best_path}")
    lo, hi = config.EXPECTED_TOP1_RANGE
    if not args.smoke and not (lo <= best_acc <= hi):
        print(f"  NOTE: outside the expected {lo:.0%}-{hi:.0%} band for a "
              f"{config.DEPTH}-block model at dim {config.EMBED_DIM}. "
              f"Worth checking before trusting the run.")


if __name__ == "__main__":
    main()