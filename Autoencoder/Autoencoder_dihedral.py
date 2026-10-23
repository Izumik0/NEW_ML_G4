"""
Autoencoder test for G-quadruplex backbone dihedrals, runs with DIFFERENT numbers of dihedrals.

GROUP_MODE = "pad"   : align columns by name (union over all runs), zero-fill + mask missing ones,
                       one autoencoder for everything. Masked entries are ignored in loss and metrics.
GROUP_MODE = "group" : runs with an identical header get their own autoencoder.
"""
import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# =====================================================================
# CONFIG
# =====================================================================
DATA_GLOB = "../dihedrals/*/bb_dihedrals.txt"   # file or pattern
STRIDE = 5                             # tells how much space is between taken frames
CACHE_DIR = Path("../cache_dihedrals")
GROUP_MODE = "pad"                      # "pad" | "group"
MIN_RUNS_PER_GROUP = 1                  # group mode: skip groups with fewer runs

SPLIT_MODE = "auto"                     # "auto" | "run" | "time" | "all"
TEST_FRAC = 0.2
TIME_GAP_FRAMES = 25

LATENT_SWEEP = [8,16]
LATENT_DETAIL = 8
HIDDEN = (256, 128)
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
SEED = 0
OUT_DIR = Path("../dihedral_ae_out")
SHOW_PLOTS = False
# =====================================================================

rng = np.random.default_rng(SEED)
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

TYPE_ORDER = ["a", "b", "g", "d", "e", "z"]


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def name_key(n):
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", n)
    if not m:
        return (10**6, 99, n)
    t = m.group(1).lower()
    return (int(m.group(2)), TYPE_ORDER.index(t) if t in TYPE_ORDER else 98, n)


def read_names(path):
    with open(path) as f:
        return f.readline().lstrip("#").split()[1:]


def load_run(path):
    key = str(Path(path).resolve()).replace(os.sep, "_").replace(":", "") + f"_s{STRIDE}.npy"
    cp = CACHE_DIR / key[-200:]
    if cp.exists():
        return np.load(cp)
    a = np.loadtxt(path, comments="#", dtype=np.float32)[::STRIDE, 1:]
    np.save(cp, a)
    return a


def to_cos_sin(deg):
    r = np.deg2rad(deg).astype(np.float32)
    return np.concatenate([np.cos(r), np.sin(r)], axis=1)


def from_cos_sin(x):
    d = x.shape[1] // 2
    return np.rad2deg(np.arctan2(x[:, d:], x[:, :d]))


def ang_err(true_deg, pred_deg):
    return np.abs((pred_deg - true_deg + 180.0) % 360.0 - 180.0)


def embed(arr, names, all_names):
    """Place a run's columns into the union layout. -> (deg (n, D) zero-filled, present (D,) bool)."""
    pos = {n: i for i, n in enumerate(all_names)}
    idx = [pos[n] for n in names]
    full = np.zeros((len(arr), len(all_names)), np.float32)
    full[:, idx] = arr
    pres = np.zeros(len(all_names), bool)
    pres[idx] = True
    return full, pres


class AE(nn.Module):
    def __init__(self, n_in, latent):
        super().__init__()
        h1, h2 = HIDDEN
        self.enc = nn.Sequential(nn.Linear(n_in, h1), nn.GELU(), nn.Linear(h1, h2), nn.GELU(),
                                 nn.Linear(h2, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h2), nn.GELU(), nn.Linear(h2, h1), nn.GELU(),
                                 nn.Linear(h1, n_in), nn.Tanh())

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


@torch.no_grad()
def reconstruct(model, X):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + 8192]).to(device))[0].cpu().numpy()
                           for i in range(0, len(X), 8192)])


def train_ae(Xtr, Mtr, Xte, Mte, latent):
    torch.manual_seed(SEED)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr)),
                    batch_size=BATCH_SIZE, shuffle=True)
    model = AE(Xtr.shape[1], latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    xte, mte = torch.from_numpy(Xte).to(device), torch.from_numpy(Mte).to(device)
    for ep in range(EPOCHS):
        model.train()
        tot = cnt = 0.0
        for xb, mb in dl:
            xb, mb = xb.to(device), mb.to(device)
            loss = (((model(xb)[0] - xb) ** 2) * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
        sched.step()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                te = ((((model(xte)[0] - xte) ** 2) * mte).sum() / mte.sum().clamp(min=1)).item()
            print(f"  latent {latent:3d} ep {ep:3d}  train {tot / cnt:.5f}  test {te:.5f}")
    return model


# ---------------------------------------------------------------------
# one full experiment on a list of (array, names) runs
# ---------------------------------------------------------------------
def experiment(run_list, tag):
    out = OUT_DIR / tag
    out.mkdir(exist_ok=True, parents=True)
    all_names = sorted({n for _, nm in run_list for n in nm}, key=name_key)
    D = len(all_names)
    emb = [embed(a, nm, all_names) for a, nm in run_list]
    print(f"\n===== {tag}: {len(run_list)} runs, {sum(len(e[0]) for e in emb)} frames, "
          f"{D} dihedral columns (union) =====")

    def stack(parts):
        deg = np.concatenate([p[0] for p in parts])
        pres = np.concatenate([np.repeat(p[1][None, :], len(p[0]), 0) for p in parts])
        return deg, pres

    mode = SPLIT_MODE
    if mode == "auto":
        mode = "run" if len(emb) >= 5 else "time"
    if mode == "all":
        tr_parts = te_parts = emb
    elif mode == "run":
        idx = rng.permutation(len(emb))
        n_test = max(1, int(round(TEST_FRAC * len(emb))))
        te_parts = [emb[i] for i in idx[:n_test]]
        tr_parts = [emb[i] for i in idx[n_test:]]
    else:  # time
        tr_parts, te_parts = [], []
        for deg, pres in emb:
            cut = int(len(deg) * (1 - TEST_FRAC))
            tr_parts.append((deg[:max(cut - TIME_GAP_FRAMES // STRIDE, 1)], pres))
            te_parts.append((deg[cut:], pres))
    print("split mode:", mode)
    tr_deg, tr_pres = stack(tr_parts)
    te_deg, te_pres = stack(te_parts)
    print(f"train frames {len(tr_deg)}, test frames {len(te_deg)}")

    Xtr, Xte = to_cos_sin(tr_deg), to_cos_sin(te_deg)
    Mtr = np.concatenate([tr_pres, tr_pres], 1).astype(np.float32)
    Mte = np.concatenate([te_pres, te_pres], 1).astype(np.float32)
    Xtr, Xte = Xtr * Mtr, Xte * Mte                     # missing entries = exactly 0

    def masked_err(pred_deg):
        e = ang_err(te_deg, pred_deg)
        e[~te_pres] = np.nan
        return e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_cs = (Xtr * Mtr).sum(0, keepdims=True) / np.maximum(Mtr.sum(0, keepdims=True), 1)
        base_err = masked_err(from_cos_sin(np.repeat(mean_cs, len(Xte), 0)))
        print(f"baseline (predict mean): mean {np.nanmean(base_err):.2f} deg, "
              f"median {np.nanmedian(base_err):.2f} deg")

        sweep = {}
        for latent in LATENT_SWEEP:
            m = train_ae(Xtr, Mtr, Xte, Mte, latent)
            e = masked_err(from_cos_sin(reconstruct(m, Xte) * Mte))
            sweep[latent] = e
            v = e[~np.isnan(e)]
            print(f"=> latent {latent:3d}: mean {v.mean():.2f}  median {np.median(v):.2f}  "
                  f"p95 {np.percentile(v, 95):.2f}  p99 {np.percentile(v, 99):.2f}\n")

        # --- figures ---
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(sweep), [np.nanmean(e) for e in sweep.values()], "o-", label="AE mean")
        ax.plot(list(sweep), [np.nanmedian(e) for e in sweep.values()], "s--", label="AE median")
        ax.axhline(np.nanmean(base_err), color="gray", ls=":", label="baseline mean")
        ax.set_xscale("log", base=2)
        ax.set(xlabel="latent size", ylabel="angular error (deg)", title=f"Error vs latent size ({tag})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "error_vs_latent.png", dpi=150)

        err = sweep.get(LATENT_DETAIL)
        if err is None:
            m = train_ae(Xtr, Mtr, Xte, Mte, LATENT_DETAIL)
            err = masked_err(from_cos_sin(reconstruct(m, Xte) * Mte))
        flat = err[~np.isnan(err)]
        bins = np.linspace(0, 180, 91)
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
        ax[0].hist(base_err[~np.isnan(base_err)], bins=bins, density=True, alpha=0.45, color="gray",
                   label=f"baseline ({np.nanmean(base_err):.1f}°)")
        for l, c in zip([min(sweep), LATENT_DETAIL, max(sweep)], ["tab:orange", "tab:blue", "tab:green"]):
            if l in sweep:
                v = sweep[l][~np.isnan(sweep[l])]
                ax[0].hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=c,
                           label=f"AE latent {l} ({v.mean():.1f}°)")
        ax[0].set(xlabel="absolute angular error (deg)", ylabel="density", title="All test values")
        ax[0].legend(fontsize=8)
        fe = np.nanmean(err, 1)
        ax[1].hist(fe, bins=80, color="tab:green", alpha=0.85)
        ax[1].axvline(np.nanmedian(fe), color="k", ls="--", label=f"median {np.nanmedian(fe):.1f}°")
        ax[1].set(xlabel="mean error per frame (deg)", ylabel="frames", title=f"Per-frame (latent {LATENT_DETAIL})")
        ax[1].legend()
        ax[2].hist(flat, bins=bins, color="tab:red", alpha=0.85)
        ax[2].set_yscale("log")
        ax[2].set(xlabel="absolute angular error (deg)", ylabel="count (log)", title="Tail")
        fig.tight_layout()
        fig.savefig(out / "error_histograms.png", dpi=150)

        per = np.nanmean(err, 0)
        order = [i for i in np.argsort(np.nan_to_num(per, nan=-1))[::-1] if not np.isnan(per[i])]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(range(len(order)), per[order], color="tab:purple")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([all_names[i] for i in order], rotation=90, fontsize=7)
        ax.set(ylabel="mean error (deg)", title=f"Error per dihedral (latent {LATENT_DETAIL})")
        fig.tight_layout()
        fig.savefig(out / "error_per_dihedral.png", dpi=150)

        print(f"latent {LATENT_DETAIL}: median {np.median(flat):.2f}°  mean {flat.mean():.2f}°  "
              f"p95 {np.percentile(flat, 95):.2f}°  p99 {np.percentile(flat, 99):.2f}°")
        print("worst dihedrals:", [all_names[i] for i in order[:5]])


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
files = sorted(glob.glob(DATA_GLOB))
if not files:
    raise FileNotFoundError(f"No files match DATA_GLOB={DATA_GLOB!r} (cwd: {os.getcwd()})")
runs = []
for f in files:
    a, nm = load_run(f), read_names(f)
    if a.shape[1] != len(nm):
        raise ValueError(f"{f}: {a.shape[1]} data columns but {len(nm)} header names")
    if np.isnan(a).any():
        raise ValueError(f"{f}: contains NaN")
    runs.append((a, nm))

counts = {}
for _, nm in runs:
    counts[len(nm)] = counts.get(len(nm), 0) + 1
print(f"{len(runs)} runs; dihedral counts per run: {dict(sorted(counts.items()))}")

if GROUP_MODE == "pad":
    experiment(runs, "all_runs_padded")
else:
    groups = {}
    for a, nm in runs:
        groups.setdefault(tuple(nm), []).append((a, nm))
    for k, (hdr, rl) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        if len(rl) < MIN_RUNS_PER_GROUP:
            print(f"skip group with {len(rl)} run(s), {len(hdr)} dihedrals")
            continue
        experiment(rl, f"group{k}_{len(hdr)}dih_{len(rl)}runs")

print("figures saved to", OUT_DIR.resolve())
if SHOW_PLOTS:
    plt.show()