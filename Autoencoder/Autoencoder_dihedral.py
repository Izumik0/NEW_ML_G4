import glob
import math
import os
import pickle
import queue
import re
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =====================================================================
# CONFIG
# =====================================================================

# Data location config
DATA_GLOB = "../dihedrals/*/bb_dihedrals.txt"
CACHE_DIR = Path("../cache_dihedrals")
OUT_DIR = Path("../dihedral_ae_out_stream")


# input data information
STRIDE = 25                      # 1 = use every 20 ps frame (whole data). Cache files are per STRIDE.
dt = 20.0                       # ps between ORIGINAL frames

# data processing config
SPLIT_MODE = "auto"             # "auto" | "run" | "time" | "all"   (see README)
TEST_FRAC = 0.15
TIME_GAP_FRAMES = 100           # time mode: gap before the test part, in ORIGINAL frames


# model config
LATENT_SWEEP = [2, 4, 8, 16, 32, 64]
LATENT_DETAIL = 8               # latent size for the detailed plots
HIDDEN = (256, 128)
BATCH_SIZE = 2048
LR = 1e-3

N_WORKERS = max(1, (os.cpu_count() or 2) - 2)   # parallel parsing of the text files (first pass only)

#  how much data each model sees
TRAIN_PASSES = 0.01              # passes over the training frames (can be fractional, e.g. 0.2)
MAX_TRAIN_FRAMES = 100_000_000  # hard cap on frames seen per model (check the printed frames/s first!)

# streaming / memory
SEG_LEN = 2048                  # contiguous frames per segment (one disk read each)
BLOCK_FRAMES = 1_000_000        # frames held in RAM at once (~ BLOCK_FRAMES x columns x 4 bytes)
EVAL_BATCH = 16384
MAX_TEST_FRAMES = 1_000_000     # test frames kept for the per-frame table/plots (thinned evenly)
BASELINE_FRAC = 0.1             # fraction of training segments used to compute the "mean" baseline

# analysis
HUMP = 16.0                     # per-frame error (deg) separating "good" and "bad" frames
TOP_PCT = 99                    # "worst frames" = above this percentile
N_WORST_RUNS = 4
RESUME = True                   # reuse finished latent sizes (models/eval files) if the config matches
VAL_BIN, FE_BIN = 0.25, 0.25    # histogram resolution in degrees
NVB, NFB = int(180 / VAL_BIN), int(180 / FE_BIN)

# other
SEED = 0
TYPE_ORDER = ["a", "b", "g", "d", "e", "z"]

# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def name_key(n):
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", n)
    if not m:
        return (10 ** 6, 99, n)
    t = m.group(1).lower()
    return (int(m.group(2)), TYPE_ORDER.index(t) if t in TYPE_ORDER else 98, n)


def read_names(path):
    with open(path) as f:
        return f.readline().lstrip("#").split()[1:]


def cache_path(path):
    key = str(Path(path).resolve()).replace(os.sep, "_").replace(":", "") + f"_s{STRIDE}.npy"
    return CACHE_DIR / key[-200:]


def ensure_cache(path):
    """Parse one run into the cache if missing; return the cached array shape. Safe against half-written files."""
    cp = cache_path(path)
    if cp.exists():
        try:
            a = np.load(cp, mmap_mode="r")
            shape = tuple(a.shape)
            del a
            return shape
        except Exception as e:
            print(f"bad cache file {cp.name} ({e}); re-parsing", flush=True)
    with open(path) as f:
        next(f)                                                  # header
        a = np.loadtxt(islice(f, 0, None, STRIDE), dtype=np.float32, ndmin=2)[:, 1:]
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError(f"{path}: no data rows after parsing")
    tmp = cp.with_name(cp.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, a)
    os.replace(tmp, cp)
    return tuple(a.shape)


def build_cache(files):
    t0, shapes = time.time(), []
    if N_WORKERS > 1:
        with ProcessPoolExecutor(N_WORKERS) as ex:
            for i, sh in enumerate(ex.map(ensure_cache, files, chunksize=2)):
                shapes.append(sh)
                if (i + 1) % 100 == 0:
                    print(f"  cache {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, f in enumerate(files):
            shapes.append(ensure_cache(f))
            if (i + 1) % 100 == 0:
                print(f"  cache {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    return shapes


def prefetch(gen, depth=2):
    """Run a generator in a background thread so disk reads overlap with training."""
    q, stop, END = queue.Queue(maxsize=depth), threading.Event(), object()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def work():
        try:
            for item in gen:
                if not put(item):
                    return
            put(END)
        except BaseException as e:           # hand the error to the consumer
            put(e)

    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def cos_sin_np(deg, pres):
    m = pres.astype(np.float32)
    r = np.deg2rad(deg).astype(np.float32)
    return np.concatenate([np.cos(r) * m, np.sin(r) * m], 1), np.concatenate([m, m], 1)


def hist_pct(H, p, binw):
    c = np.cumsum(H)
    tot = c[-1]
    k = int(np.searchsorted(c, p / 100 * tot))
    k = min(k, len(H) - 1)
    prev = c[k - 1] if k > 0 else 0.0
    return (k + (p / 100 * tot - prev) / max(H[k], 1.0)) * binw


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# data layout shared by all stages (filled in main)
# ---------------------------------------------------------------------
class Data:
    files = []          # cache file per run (.npy)
    col_idx = []        # per run: positions of its columns in the union layout
    pres = None         # (n_runs, D) bool
    D = 0


def load_block(segs):
    """segs: list of (run, start, stop, step). -> deg (N, D) float32 zero-padded, run ids, frame positions."""
    segs = sorted(segs)
    n = sum(len(range(s, e, st)) for _, s, e, st in segs)
    B = np.zeros((n, Data.D), np.float32)
    rid = np.empty(n, np.int32)
    tpos = np.empty(n, np.int64)
    o = 0
    for ri, s, e, st in segs:
        a = np.load(Data.files[ri], mmap_mode="r")
        seg = np.array(a[s:e:st])                      # real copy, then release the mapping
        del a
        if not np.isfinite(seg).all():
            raise ValueError(f"non-finite values in {Data.files[ri]}")
        k = len(seg)
        B[o:o + k, Data.col_idx[ri]] = seg
        rid[o:o + k] = ri
        tpos[o:o + k] = np.arange(s, e, st)[:k]
        o += k
    return B, rid, tpos


def block_iter(segs, block_frames):
    cur, n = [], 0
    for sg in segs:
        cur.append(sg)
        n += len(range(sg[1], sg[2], sg[3]))
        if n >= block_frames:
            yield load_block(cur)
            cur, n = [], 0
    if cur:
        yield load_block(cur)


def make_segments(ranges, step=1):
    out = []
    for ri, s, e in ranges:
        for s0 in range(s, e, SEG_LEN * step):
            out.append((ri, s0, min(e, s0 + SEG_LEN * step), step))
    return out


# ---------------------------------------------------------------------
# baseline and training
# ---------------------------------------------------------------------
def baseline_mean_cs(train_segs, rng):
    k = max(1, int(len(train_segs) * BASELINE_FRAC))
    sel = [train_segs[i] for i in rng.choice(len(train_segs), k, replace=False)]
    xs, ms = np.zeros(2 * Data.D), np.zeros(2 * Data.D)
    for B, rid, _ in prefetch(block_iter(sel, BLOCK_FRAMES)):
        for i in range(0, len(B), EVAL_BATCH):
            x, m = cos_sin_np(B[i:i + EVAL_BATCH], Data.pres[rid[i:i + EVAL_BATCH]])
            xs += x.sum(0, dtype=np.float64)
            ms += m.sum(0, dtype=np.float64)
    return (xs / np.maximum(ms, 1)).astype(np.float32)[None, :]


def train_ae(latent, train_segs, n_train, device, pres_t, rng):
    if n_train < BATCH_SIZE:
        raise ValueError("fewer training frames than BATCH_SIZE")
    torch.manual_seed(SEED)
    budget = int(min(TRAIN_PASSES * n_train, MAX_TRAIN_FRAMES))
    total_steps = max(budget // BATCH_SIZE, 1)
    print(f"  latent {latent}: {budget:,} frames = {budget / n_train:.2f} passes, {total_steps:,} steps", flush=True)
    model = AE(2 * Data.D, latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step, run_loss, run_n, t0 = 0, torch.zeros((), device=device), 0, time.time()
    every = max(total_steps // 20, 1)
    while step < total_steps:
        order = list(train_segs)
        rng.shuffle(order)
        for B, rid, _ in prefetch(block_iter(order, BLOCK_FRAMES)):
            Bt = torch.from_numpy(B).to(device)
            rt = torch.from_numpy(rid.astype(np.int64)).to(device)
            perm = torch.from_numpy(rng.permutation(len(B))).to(device)
            model.train()
            for i in range(0, len(B) - BATCH_SIZE + 1, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                m = pres_t[rt[idx]].float()
                rad = Bt[idx] * (math.pi / 180.0)
                x = torch.cat([torch.cos(rad) * m, torch.sin(rad) * m], 1)
                mm = torch.cat([m, m], 1)
                loss = (((model(x)[0] - x) ** 2) * mm).sum() / mm.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                run_loss += loss.detach()
                run_n += 1
                if step % every == 0 or step == total_steps:
                    el = time.time() - t0
                    fps = step * BATCH_SIZE / el
                    eta = (total_steps - step) * BATCH_SIZE / fps
                    print(f"    step {step:,}/{total_steps:,}  loss {run_loss.item() / run_n:.5f}  "
                          f"{fps:,.0f} frames/s  ETA {eta / 60:.1f} min", flush=True)
                    run_loss.zero_(); run_n = 0
                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    return model


def make_predict(model, device):
    model.eval()

    @torch.no_grad()
    def f(x):
        return model(torch.from_numpy(np.ascontiguousarray(x)).to(device))[0].cpu().numpy()
    return f


# ---------------------------------------------------------------------
# streaming evaluation
# ---------------------------------------------------------------------
def evaluate(predict_np, test_segs, n_runs, K=20):
    D = Data.D
    H, F = np.zeros(NVB), np.zeros(NFB)
    S, C = np.zeros((NFB, D)), np.zeros((NFB, D))
    run_sum, run_cnt = np.zeros(n_runs), np.zeros(n_runs)
    fe_l, rid_l, t_l = [], [], []
    top_fe, top_rows, top_rid, top_t = np.empty(0), np.empty((0, D)), np.empty(0, int), np.empty(0)
    for B, rid, tpos in prefetch(block_iter(test_segs, BLOCK_FRAMES)):
        for i in range(0, len(B), EVAL_BATCH):
            deg, r = B[i:i + EVAL_BATCH], rid[i:i + EVAL_BATCH]
            pres = Data.pres[r]
            x, m = cos_sin_np(deg, pres)
            xh = predict_np(x) * m
            pred = np.degrees(np.arctan2(xh[:, D:], xh[:, :D]))
            pf = pres.astype(np.float32)
            err = np.abs((pred - deg + 180.0) % 360.0 - 180.0) * pf
            fe = err.sum(1) / pf.sum(1)
            vals = err[pres]
            H += np.bincount(np.minimum((vals / VAL_BIN).astype(np.int64), NVB - 1), minlength=NVB)
            fb = np.minimum((fe / FE_BIN).astype(np.int64), NFB - 1)
            F += np.bincount(fb, minlength=NFB)
            for d in range(D):
                S[:, d] += np.bincount(fb, weights=err[:, d], minlength=NFB)
                C[:, d] += np.bincount(fb, weights=pf[:, d], minlength=NFB)
            run_sum += np.bincount(r, weights=fe, minlength=n_runs)
            run_cnt += np.bincount(r, minlength=n_runs)
            fe_l.append(fe.astype(np.float32)); rid_l.append(r.copy())
            t_l.append(tpos[i:i + EVAL_BATCH].astype(np.float64) * STRIDE * dt)
            # running top-K worst frames
            kk = min(K, len(fe))
            ci = np.argpartition(fe, -kk)[-kk:]
            top_fe = np.concatenate([top_fe, fe[ci]]); top_rows = np.concatenate([top_rows, err[ci]])
            top_rid = np.concatenate([top_rid, r[ci]]); top_t = np.concatenate([top_t, t_l[-1][ci]])
            keep = np.argsort(top_fe)[::-1][:K]
            top_fe, top_rows, top_rid, top_t = top_fe[keep], top_rows[keep], top_rid[keep], top_t[keep]
    return dict(H=H, F=F, S=S, C=C, run_sum=run_sum, run_cnt=run_cnt,
                fe=np.concatenate(fe_l), rid=np.concatenate(rid_l), t_ps=np.concatenate(t_l),
                top_fe=top_fe, top_rows=top_rows, top_rid=top_rid, top_t=top_t)


def summarize(r):
    return dict(mean=r["S"].sum() / r["C"].sum(), median=hist_pct(r["H"], 50, VAL_BIN),
                p95=hist_pct(r["H"], 95, VAL_BIN), p99=hist_pct(r["H"], 99, VAL_BIN))


# ---------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------
def report(results, latents, all_names, labels, n_dih_run, out):
    names_arr = np.array(all_names)
    base = results["baseline"]
    stats = {k: summarize(results[k]) for k in ["baseline"] + latents}
    print("\n==== summary (test error, degrees; median/p95/p99 from 0.25-degree histograms) ====")
    print(pd.DataFrame(stats).T.round(2).to_string())

    # error vs latent
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(latents, [stats[l]["mean"] for l in latents], "o-", label="AE mean")
    ax.plot(latents, [stats[l]["median"] for l in latents], "s--", label="AE median")
    ax.axhline(stats["baseline"]["mean"], color="gray", ls=":", label="baseline mean")
    ax.set_xscale("log", base=2)
    ax.set(xlabel="latent size", ylabel="angular error (deg)", title="Error vs latent size")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "error_vs_latent.png", dpi=150); plt.close(fig)

    detail = min(latents, key=lambda l: abs(l - LATENT_DETAIL))
    rd = results[detail]
    centers_v = (np.arange(NVB) + 0.5) * VAL_BIN
    centers_f = (np.arange(NFB) + 0.5) * FE_BIN

    # histograms
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].fill_between(centers_v, base["H"] / base["H"].sum() / VAL_BIN, step="mid", color="gray", alpha=0.45,
                       label=f"baseline ({stats['baseline']['mean']:.1f}°)")
    for l, c in zip([min(latents), detail, max(latents)], ["tab:orange", "tab:blue", "tab:green"]):
        ax[0].step(centers_v, results[l]["H"] / results[l]["H"].sum() / VAL_BIN, where="mid", color=c, lw=1.6,
                   label=f"AE latent {l} ({stats[l]['mean']:.1f}°)")
    ax[0].set(xlabel="absolute angular error (deg)", ylabel="density", title="All test values", xlim=(0, 180))
    ax[0].legend(fontsize=8)
    ax[1].bar(centers_f, rd["F"], width=FE_BIN, color="tab:green", alpha=0.85)
    med = hist_pct(rd["F"], 50, FE_BIN)
    ax[1].axvline(med, color="k", ls="--", label=f"median {med:.1f}°")
    nz = np.nonzero(rd["F"])[0]
    ax[1].set(xlabel="mean error per frame (deg)", ylabel="frames", title=f"Per-frame (latent {detail})",
              xlim=(0, centers_f[nz.max()] * 1.05))
    ax[1].legend()
    ax[2].bar(centers_v, rd["H"], width=VAL_BIN, color="tab:red", alpha=0.85)
    ax[2].set_yscale("log")
    ax[2].set(xlabel="absolute angular error (deg)", ylabel="count (log)", title="Tail", xlim=(0, 180))
    fig.tight_layout(); fig.savefig(out / "error_histograms.png", dpi=150); plt.close(fig)

    # per dihedral
    per = rd["S"].sum(0) / np.where(rd["C"].sum(0) > 0, rd["C"].sum(0), np.nan)
    order = [i for i in np.argsort(np.nan_to_num(per, nan=-1))[::-1] if not np.isnan(per[i])]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(order)), per[order], color="tab:purple")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([all_names[i] for i in order], rotation=90, fontsize=7)
    ax.set(ylabel="mean error (deg)", title=f"Error per dihedral (latent {detail})")
    fig.tight_layout(); fig.savefig(out / "error_per_dihedral.png", dpi=150); plt.close(fig)

    # per-frame table (thinned sample of test frames, identical order for every latent size)
    cols = [f"err_L{l}" for l in latents]
    df = pd.DataFrame({"run": labels[base["rid"]], "t_ns": base["t_ps"] / 1000.0,
                       "n_dihedrals": n_dih_run[base["rid"]]})
    for l, c in zip(latents, cols):
        df[c] = results[l]["fe"]
    df.to_csv(out / "frame_errors_all_latents.csv", index=False, float_format="%.3f")

    top_masks = {}
    for l, c in zip(latents, cols):
        r = results[l]
        print(f"\n================ latent {l} ================")
        rows = []
        for fe_, row, ri, t in zip(r["top_fe"][:20], r["top_rows"][:20], r["top_rid"][:20], r["top_t"][:20]):
            rows.append(dict(run=labels[ri], t_ns=t / 1000.0, frame_err=fe_,
                             worst_3_dihedrals=", ".join(names_arr[np.argsort(row)[::-1][:3]])))
        print("20 worst frames over the WHOLE test set:\n", pd.DataFrame(rows).round(2).to_string(index=False))
        kh = int(HUMP / FE_BIN)
        print(f"\n{r['F'][kh:].sum() / r['F'].sum():.1%} of test frames above {HUMP} deg; top runs there "
              f"(in the thinned sample):\n{df[df[c] > HUMP]['run'].value_counts().head(8)}")
        k99 = min(int(np.searchsorted(np.cumsum(r["F"]), TOP_PCT / 100 * r["F"].sum())), NFB - 1)
        cs = r["C"][k99:].sum(0)
        extra = pd.Series(np.where(cs > 0, r["S"][k99:].sum(0) / np.maximum(cs, 1), np.nan)
                          - r["S"].sum(0) / np.maximum(r["C"].sum(0), 1), index=all_names).sort_values(ascending=False)
        print(f"\ndihedrals that are extra bad in the worst {100 - TOP_PCT}% of frames:\n{extra.head(10)}")
        top_masks[l] = df[c].values >= np.nanpercentile(df[c].values, TOP_PCT)

    # per run (exact: all evaluated frames, not the sample)
    seen = base["run_cnt"] > 0
    per_run = pd.DataFrame({f"err_L{l}": results[l]["run_sum"][seen] / results[l]["run_cnt"][seen] for l in latents},
                           index=labels[seen])
    per_run["n_dih"] = n_dih_run[seen]; per_run["frames"] = base["run_cnt"][seen].astype(int)
    per_run = per_run.sort_values(cols[-1], ascending=False)
    per_run.to_csv(out / "per_run_error_all_latents.csv")
    print("\nper-run mean error (worst 15):\n", per_run.head(15).round(2).to_string())
    print("\nmean error by number of dihedrals:\n", df.groupby("n_dihedrals")[cols].mean().round(2).to_string())

    ov = pd.DataFrame(index=latents, columns=latents, dtype=float)
    for a in latents:
        for b in latents:
            ov.loc[a, b] = (top_masks[a] & top_masks[b]).sum() / max((top_masks[a] | top_masks[b]).sum(), 1)
    print(f"\noverlap (Jaccard) of the worst {100 - TOP_PCT}% frames between latent sizes:\n", ov.round(2).to_string())

    # worst runs over time
    worst_runs = per_run[cols].mean(1).sort_values(ascending=False).index[:N_WORST_RUNS]
    fig, axs = plt.subplots(len(worst_runs), 1, figsize=(11, 2.8 * len(worst_runs)), squeeze=False)
    for ax, r_ in zip(axs[:, 0], worst_runs):
        d = df[df.run == r_]
        for l in latents:
            ax.plot(d.t_ns, d[f"err_L{l}"], lw=0.8, label=f"latent {l}")
        ax.set(ylabel="deg", title=f"run {r_}")
    axs[0, 0].legend(fontsize=7, ncol=len(latents)); axs[-1, 0].set_xlabel("time (ns)")
    fig.suptitle("Mean error per frame over time (thinned test sample)")
    fig.tight_layout(); fig.savefig(out / "worst_runs_error_vs_time_all_latents.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    hi = max(centers_f[np.nonzero(results[l]["F"])[0].max()] for l in latents)
    for l in latents:
        ax.step(centers_f, results[l]["F"], where="mid", lw=1.5, label=f"latent {l}")
    ax.set(xlabel="mean error per frame (deg)", ylabel="frames", xlim=(0, hi * 1.05),
           title="Per-frame error, all latent sizes")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "per_frame_hist_all_latents.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No files match DATA_GLOB={DATA_GLOB!r} (cwd: {os.getcwd()})")
    print(f"{len(files)} runs; building/validating cache (STRIDE={STRIDE}, {N_WORKERS} workers)")
    shapes = build_cache(files)
    lengths = [s[0] for s in shapes]
    names_list = [read_names(f) for f in files]
    for f, sh, nm in zip(files, shapes, names_list):
        if sh[1] != len(nm):
            raise ValueError(f"{f}: {sh[1]} data columns but {len(nm)} header names")
    all_names = sorted({n for nm in names_list for n in nm}, key=name_key)
    D, n_runs = len(all_names), len(files)
    pos = {n: i for i, n in enumerate(all_names)}
    Data.D = D
    Data.files = [str(cache_path(f)) for f in files]
    Data.col_idx = [np.array([pos[n] for n in nm]) for nm in names_list]
    Data.pres = np.zeros((n_runs, D), bool)
    for i, ix in enumerate(Data.col_idx):
        Data.pres[i, ix] = True
    n_dih_run = Data.pres.sum(1)
    labels = np.array([Path(f).parent.name or Path(f).stem for f in files], dtype=object)
    if len(set(labels)) < n_runs:
        print("WARNING: run folder names are not unique; per-run tables will merge identically named runs")
    total_frames = sum(lengths)
    print(f"{total_frames:,} cached frames in total, {D} dihedral columns (union); "
          f"dihedral counts per run: {dict(sorted(pd.Series(n_dih_run).value_counts().items()))}")
    print(f"~{total_frames * D * 4 / 1e9:.0f} GB if everything were in RAM as float32; "
          f"this script keeps about {BLOCK_FRAMES * D * 4 * 4 / 1e9:.1f} GB at a time")

    # ---- split ----
    mode = SPLIT_MODE if SPLIT_MODE != "auto" else ("run" if n_runs >= 5 else "time")
    if mode == "all":
        tr_r = te_r = [(i, 0, L) for i, L in enumerate(lengths)]
    elif mode == "run":
        perm = rng.permutation(n_runs)
        n_test = max(1, int(round(TEST_FRAC * n_runs)))
        te_r = [(int(i), 0, lengths[i]) for i in sorted(perm[:n_test])]
        tr_r = [(int(i), 0, lengths[i]) for i in sorted(perm[n_test:])]
    elif mode == "time":
        tr_r, te_r = [], []
        gap = max(TIME_GAP_FRAMES // STRIDE, 1)
        for i, L in enumerate(lengths):
            cut = int(L * (1 - TEST_FRAC))
            tr_r.append((i, 0, max(cut - gap, 1)))
            te_r.append((i, cut, L))
    else:
        raise ValueError(f"unknown SPLIT_MODE {SPLIT_MODE!r}")
    n_train = sum(e - s for _, s, e in tr_r)
    n_test_all = sum(e - s for _, s, e in te_r)
    test_every = max(1, math.ceil(n_test_all / MAX_TEST_FRAMES))
    print(f"split mode {mode}: {n_train:,} train frames, {n_test_all:,} test frames "
          f"(evaluating every {test_every}th -> ~{n_test_all // test_every:,})")
    train_segs = make_segments(tr_r)
    test_segs = make_segments(te_r, step=test_every)

    sig = repr(dict(STRIDE=STRIDE, TRAIN_PASSES=TRAIN_PASSES, MAX_TRAIN_FRAMES=MAX_TRAIN_FRAMES, SEED=SEED,
                    HIDDEN=HIDDEN, BATCH_SIZE=BATCH_SIZE, LR=LR, n_runs=n_runs, D=D, mode=mode,
                    TEST_FRAC=TEST_FRAC, test_every=test_every, SEG_LEN=SEG_LEN))
    out = OUT_DIR / "all_runs_streamed"
    (out / "models").mkdir(parents=True, exist_ok=True)

    def load_eval(tag):
        p = out / f"eval_{tag}.pkl"
        if RESUME and p.exists():
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            if d.get("sig") == sig:
                return d["res"]
        return None

    def save_eval(tag, res):
        with open(out / f"eval_{tag}.pkl", "wb") as fh:
            pickle.dump(dict(sig=sig, res=res), fh)

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results["baseline"] = load_eval("baseline")
        if results["baseline"] is None:
            print("baseline (predict mean) ...", flush=True)
            mean_cs = baseline_mean_cs(train_segs, rng)
            results["baseline"] = evaluate(lambda x: np.broadcast_to(mean_cs, x.shape), test_segs, n_runs)
            save_eval("baseline", results["baseline"])
        pres_t = torch.from_numpy(Data.pres).to(device)
        for latent in LATENT_SWEEP:
            res = load_eval(f"L{latent}")
            if res is None:
                print(f"\ntraining latent {latent}", flush=True)
                model = train_ae(latent, train_segs, n_train, device, pres_t, rng)
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "latent": latent, "n_in": 2 * D, "hidden": HIDDEN, "all_names": all_names,
                            "stride": STRIDE}, out / "models" / f"ae_latent{latent}.pt")
                res = evaluate(make_predict(model, device), test_segs, n_runs)
                save_eval(f"L{latent}", res)
            else:
                print(f"latent {latent}: reusing finished result")
            results[latent] = res
            s = summarize(res)
            print(f"=> latent {latent:3d}: mean {s['mean']:.2f}  median {s['median']:.2f}  "
                  f"p95 {s['p95']:.2f}  p99 {s['p99']:.2f}", flush=True)
        report(results, list(LATENT_SWEEP), all_names, labels, n_dih_run, out)

    print("\nresults saved to", out.resolve())


if __name__ == "__main__":
    main()