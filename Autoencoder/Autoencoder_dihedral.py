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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# =====================================================================
# CONFIG
# =====================================================================
DATA_GLOB = "../dihedrals/*/bb_dihedrals.txt"   # file or pattern
STRIDE = 25                             # tells how much space is between taken frames
dt = 20.0                              # time step in calc
CACHE_DIR = Path("cache_dihedrals")
GROUP_MODE = "pad"                      # "pad" | "group"
MIN_RUNS_PER_GROUP = 1                  # group mode: skip groups with fewer runs

SPLIT_MODE = "auto"                     # "auto" | "run" | "time" | "all"
TEST_FRAC = 0.2
TIME_GAP_FRAMES = 25

LATENT_SWEEP = [2,4,8,16,32,64]
LATENT_DETAIL = 32
HIDDEN = (256, 128)
EPOCHS = 15
BATCH_SIZE = 512
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
    all_names = sorted({n for _, nm, _ in run_list for n in nm}, key=name_key)
    D = len(all_names)
    emb = [(*embed(a, nm, all_names), f, 0) for a, nm, f in run_list]
    print(f"\n===== {tag}: {len(run_list)} runs, {sum(len(e[0]) for e in emb)} frames, "
          f"{D} dihedral columns (union) =====")

    def stack(parts):
        deg = np.concatenate([p[0] for p in parts])
        pres = np.concatenate([np.repeat(p[1][None, :], len(p[0]), 0) for p in parts])
        run = np.concatenate([np.full(len(p[0]), Path(p[2]).parent.name or Path(p[2]).name, dtype=object) for p in parts])
        t_ps = np.concatenate([(np.arange(len(p[0]))+p[3])*STRIDE*dt for p in parts])
        return deg, pres, run, t_ps

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
        for deg, pres, f, _ in emb:
            cut = int(len(deg) * (1 - TEST_FRAC))
            tr_parts.append((deg[:max(cut - TIME_GAP_FRAMES // STRIDE, 1)], pres, f, 0))
            te_parts.append((deg[cut:], pres, f, cut))
    print("split mode:", mode)
    tr_deg, tr_pres, _, _ = stack(tr_parts)
    te_deg, te_pres, te_run, te_t = stack(te_parts)
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
            torch.save({
                "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},  # CPU tensors, loadable anywhere
                "latent": latent,
                "n_in": Xtr.shape[1],
                "hidden": HIDDEN,
                "all_names": all_names,  # column order the model expects
                "stride": STRIDE,
                "epochs": EPOCHS,
            }, out / f"ae_dih_l{latent}.pt")
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
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        ax.hist(base_err[~np.isnan(base_err)], bins=bins, density=True, alpha=0.45, color="gray",
                   label=f"baseline ({np.nanmean(base_err):.1f}°)")
        for l, c in zip([min(sweep), LATENT_DETAIL, max(sweep)], ["tab:orange", "tab:blue", "tab:green"]):
            if l in sweep:
                v = sweep[l][~np.isnan(sweep[l])]
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=c,
                           label=f"AE latent {l} ({v.mean():.1f}°)")
        ax.set(xlabel="absolute angular error (deg)", ylabel="density", title="All test values")
        ax.legend(fontsize=8)
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

        def frame_analysis(sweep, te_run, te_t, all_names, out, hump=16.0, top_pct=99, n_worst_runs=4):
            latents = sorted(sweep)
            names_arr = np.array(all_names)

            # one row per test frame, one error column per latent size
            df = pd.DataFrame({"run": te_run, "t_ns": te_t / 1000.0,
                               "n_dihedrals": (~np.isnan(sweep[latents[0]])).sum(1)})
            for l in latents:
                df[f"err_L{l}"] = np.nanmean(sweep[l], 1)
            df.to_csv(out / "frame_errors_all_latents.csv", index=False)
            cols = [f"err_L{l}" for l in latents]

            # ---- per latent: worst frames, hump, dihedrals behind the worst 1% ----
            top_masks = {}
            for l in latents:
                e, c = sweep[l], f"err_L{l}"
                print(f"\n================ latent {l} ================")
                worst = df.sort_values(c, ascending=False).head(20).copy()
                rows = np.nan_to_num(e[worst.index.values], nan=-1)
                worst["worst_3_dihedrals"] = [", ".join(names_arr[np.argsort(r)[::-1][:3]]) for r in rows]
                print(worst[["run", "t_ns", c, "worst_3_dihedrals"]].to_string(index=False))

                h = df[df[c] > hump]
                print(f"\n{len(h) / len(df):.1%} of frames above {hump} deg; top runs there:\n"
                      f"{h['run'].value_counts().head(8)}")

                top = df[c].values >= np.nanpercentile(df[c].values, top_pct)
                top_masks[l] = top
                extra = pd.Series(np.nanmean(e[top], 0) - np.nanmean(e, 0), index=all_names).sort_values(
                    ascending=False)
                print(f"\ndihedrals that are extra bad in the worst {100 - top_pct}% of frames:\n{extra.head(10)}")

            # ---- per run: error for every latent side by side ----
            per_run = df.groupby("run")[cols].mean()
            per_run["n_dih"] = df.groupby("run")["n_dihedrals"].first()
            per_run["frames"] = df.groupby("run").size()
            per_run = per_run.sort_values(cols[-1], ascending=False)
            per_run.to_csv(out / "per_run_error_all_latents.csv")
            print("\nper-run mean error (worst 15 by the largest latent):\n", per_run.head(15).round(2).to_string())
            print("\nmean error by number of dihedrals:\n", df.groupby("n_dihedrals")[cols].mean().round(2).to_string())

            # ---- do different latent sizes fail on the same frames? ----
            ov = pd.DataFrame(index=latents, columns=latents, dtype=float)
            for a in latents:
                for b in latents:
                    inter = (top_masks[a] & top_masks[b]).sum()
                    ov.loc[a, b] = inter / max((top_masks[a] | top_masks[b]).sum(), 1)
            print(f"\noverlap (Jaccard) of the worst {100 - top_pct}% frames between latent sizes:\n",
                  ov.round(2).to_string())

            # ---- error vs time in the worst runs, one line per latent ----
            worst_runs = df.groupby("run")[cols].mean().mean(1).sort_values(ascending=False).index[:n_worst_runs]
            fig, axs = plt.subplots(len(worst_runs), 1, figsize=(11, 2.8 * len(worst_runs)), sharex=False,
                                    squeeze=False)
            for ax, r in zip(axs[:, 0], worst_runs):
                d = df[df.run == r]
                for l in latents:
                    ax.plot(d.t_ns, d[f"err_L{l}"], lw=0.8, label=f"latent {l}")
                ax.set(ylabel="deg", title=f"run {r}")
            axs[0, 0].legend(fontsize=7, ncol=len(latents))
            axs[-1, 0].set_xlabel("time (ns)")
            fig.suptitle("Mean error per frame over time, all latent sizes")
            fig.tight_layout()
            fig.savefig(out / "worst_runs_error_vs_time_all_latents.png", dpi=150)

            # ---- error histogram of frame errors, all latents on one axis ----
            fig, ax = plt.subplots(figsize=(7, 4))
            hi = max(np.nanmax(df[c]) for c in cols)
            b = np.linspace(0, hi, 80)
            for l in latents:
                ax.hist(df[f"err_L{l}"].dropna(), bins=b, histtype="step", lw=1.5, label=f"latent {l}")
            ax.set(xlabel="mean error per frame (deg)", ylabel="frames", title="Per-frame error, all latent sizes")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "per_frame_hist_all_latents.png", dpi=150)
            return df, per_run

        df_frames, per_run = frame_analysis(sweep, te_run, te_t, all_names, out)


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
    runs.append((a, nm, f))

counts = {}
for _, nm, _ in runs:
    counts[len(nm)] = counts.get(len(nm), 0) + 1
print(f"{len(runs)} runs; dihedral counts per run: {dict(sorted(counts.items()))}")

if GROUP_MODE == "pad":
    experiment(runs, "all_runs_padded")
else:
    groups = {}
    for a, nm, f in runs:
        groups.setdefault(tuple(nm), []).append((a, nm,f))
    for k, (hdr, rl) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        if len(rl) < MIN_RUNS_PER_GROUP:
            print(f"skip group with {len(rl)} run(s), {len(hdr)} dihedrals")
            continue
        experiment(rl, f"group{k}_{len(hdr)}dih_{len(rl)}runs")

print("figures saved to", OUT_DIR.resolve())
if SHOW_PLOTS:
    plt.show()