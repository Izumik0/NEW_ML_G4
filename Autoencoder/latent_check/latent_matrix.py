"""
Analysis of the autoencoder latent space (notebook friendly).

Needs, from a finished run of Autoencoder_matrix.py:
  <results_dir>/ae_L<L>.pt            model weights and SCALE
  <results_dir>/frame_errors.csv.gz   the test frames (run, position, split, group, time) and their errors
  <cache_dir>/distance_arr_*.npy      the cache of upper-triangle distance vectors

Nothing is trained or converted here: the saved test frames are encoded with the saved model.

Notebook use:
    from latent_tools import LatentSpace
    ls = LatentSpace("../Autoencoder_results_v3", "cache", latent=16, csv="../lifetimes_data_simple.csv")
    # add skip_missing=True if some runs have no cache file
    ls.stats()                                  # how many dimensions are used
    ls.plot_stats()
    ls.variance_decomposition()                 # what each dimension encodes (run, condition, time ...)
    ls.correlations(); ls.plot_correlations()   # link to Rg, mean distance, time, time to unfolding
    ls.probe()                                  # how well the code predicts those quantities (train -> held-out)
    ls.plot_embedding("group"); ls.plot_embedding("split"); ls.plot_embedding("rg")
    ls.plot_traverse("pc", 0)                   # what moving along a direction does to the matrix
    ls.plot_trajectories(); ls.plot_run_timeseries()
    ls.plot_novelty()                           # are held-out sequences covered by the training latents?
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

N = 158                                   # matrix size
DESC = ["rg", "mean_d", "max_d", "std_d"]  # descriptors computed from each distance vector (nm)


# ---------------- model: must be identical to the training script ----------------
def blok(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU())


def up(i, o):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                         nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class ConvEncoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.encoder = nn.Sequential(blok(1, 16), blok(16, 32), blok(32, 64), blok(64, 128))
        self.to_z = nn.Linear(128 * 10 * 10, latent)
        self.from_z = nn.Linear(latent, 128 * 10 * 10)
        self.decoder = nn.Sequential(up(128, 64), up(64, 32), up(32, 16),
                                     nn.Upsample(scale_factor=2, mode="nearest"),
                                     nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        z = self.to_z(self.encoder(x).flatten(1))
        out = self.decoder(self.from_z(z).view(-1, 128, 10, 10))[:, :, 1:-1, 1:-1]
        return 0.5 * (out + out.transpose(2, 3)), z


def _finish(fig, save, show):
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()


# ---------------- the analysis object ----------------
class LatentSpace:
    def __init__(self, results_dir, cache_dir, latent, csv=None, device=None,
                 batch=512, max_frames=None, seed=0, force=False, skip_missing=False):
        """
        results_dir : folder with ae_L<latent>.pt and frame_errors.csv.gz
        cache_dir   : folder with the distance_arr_*.npy cache
        latent      : bottleneck size of the model to analyse
        csv         : lifetimes table (optional); adds unf_ns and tte_ns (time to unfolding, group B)
        max_frames  : encode only a random subset of the test frames (faster)
        force       : re-encode even if latents_L*.npz already exists
        skip_missing: if some runs have no cache file, analyse only the runs that do (default: raise an error)
        """
        self.res, self.cache, self.L, self.batch = Path(results_dir), Path(cache_dir), latent, batch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.dev = torch.device(device)
        ck = torch.load(self.res / f"ae_L{latent}.pt", map_location=self.dev)
        self.scale = float(ck["scale"])
        self.model = ConvEncoder(latent).to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self._iu = torch.triu_indices(N, N, 1).to(self.dev)

        m = pd.read_csv(self.res / "frame_errors.csv.gz")
        if max_frames and len(m) > max_frames:
            m = m.sample(max_frames, random_state=seed).sort_index()
        m = m.reset_index(drop=True)
        run_key = (m["seq"].astype(str) + "_" + m["top"].astype(str) + "_"
                   + m["lay"].astype(str) + "_" + m["copy"].astype(str))
        have = {p.stem[len("distance_arr_"):] for p in self.cache.glob("distance_arr_*.npy")}
        ok = run_key.isin(have)
        if not ok.all():
            miss = run_key[~ok].nunique()
            if not skip_missing:
                raise FileNotFoundError(
                    f"{miss} runs ({int((~ok).sum())} test frames) have no cache file in {self.cache}, "
                    f"e.g. {run_key[~ok].iloc[0]}. Use skip_missing=True to analyse only the runs that have one.")
            print(f"skipping {miss} runs without a cache file ({(~ok).mean():.1%} of the test frames)")
            m = m[ok].reset_index(drop=True)
        if len(m) == 0:
            raise FileNotFoundError(f"no cache files from frame_errors.csv.gz were found in {self.cache}")
        meta = m[["pos", "seq", "top", "lay", "copy", "group", "split", "t_ns"]].copy()
        if f"enc_{latent}" in m:
            meta["err_nm"] = np.sqrt(m[f"enc_{latent}"].values) * self.scale     # per-frame RMSE in nm
        meta["run"] = (meta["seq"].astype(str) + "_" + meta["top"].astype(str) + "_"
                       + meta["lay"].astype(str) + "_" + meta["copy"].astype(str))
        if csv is not None:
            life = pd.read_csv(csv, index_col=0)[["seq", "top", "lay", "copy", "unf_time"]]
            meta = meta.merge(life, on=["seq", "top", "lay", "copy"], how="left")
            meta["unf_ns"] = meta["unf_time"] / 1000
            meta["tte_ns"] = np.where(meta["group"] == "B", meta["unf_ns"] - meta["t_ns"], np.nan)
        assert len(meta) == len(m), "lifetimes table has duplicate rows for some runs"
        self.meta = meta
        self._emb = {}

        f = self.res / f"latents_L{latent}_{len(meta)}_s{seed}{'_sm' if skip_missing else ''}.npz"
        if f.exists() and not force:
            d = np.load(f)
            self.Z = d["Z"]
            for c in DESC:
                self.meta[c] = d[c]
            print(f"loaded {len(self.Z)} encoded frames from {f.name}")
        else:
            self._encode()
            np.savez_compressed(f, Z=self.Z, **{c: self.meta[c].values for c in DESC})
            print(f"saved {f.name}")

    # ---------- encoding ----------
    def _encode(self):
        meta, n = self.meta, len(self.meta)
        Z = np.zeros((n, self.L), np.float32)
        desc = {c: np.zeros(n, np.float32) for c in DESC}

        def process(rows, V):                       # V: (b, 12403) distances in nm
            x = torch.from_numpy(V).to(self.dev) / self.scale
            full = torch.zeros(len(V), N, N, device=self.dev)
            full[:, self._iu[0], self._iu[1]] = x
            full = (full + full.transpose(1, 2)).unsqueeze(1)
            with torch.no_grad():
                z = self.model.to_z(self.model.encoder(F.pad(full, (1, 1, 1, 1), mode="replicate")).flatten(1))
            Z[rows] = z.cpu().numpy()
            desc["rg"][rows] = np.sqrt((V ** 2).sum(1)) / N          # radius of gyration of the 158 points
            desc["mean_d"][rows] = V.mean(1)
            desc["max_d"][rows] = V.max(1)
            desc["std_d"][rows] = V.std(1)

        pos_all = meta["pos"].values
        rows_b, V_b = [], []
        groups = meta.groupby("run").indices
        for i, (run, rows) in enumerate(groups.items()):
            r0 = meta.iloc[rows[0]]
            path = self.cache / f"distance_arr_{r0['seq']}_{r0['top']}_{r0['lay']}_{r0['copy']}.npy"
            arr = np.load(path, mmap_mode="r")
            order = np.argsort(pos_all[rows])
            rows = rows[order]
            rows_b.append(rows)
            V_b.append(np.asarray(arr[pos_all[rows]], dtype=np.float32))
            if sum(len(r) for r in rows_b) >= self.batch:
                process(np.concatenate(rows_b), np.concatenate(V_b))
                rows_b, V_b = [], []
            if i % 500 == 0:
                print(f"  encoded {i}/{len(groups)} runs", flush=True)
        if rows_b:
            process(np.concatenate(rows_b), np.concatenate(V_b))
        self.Z = Z
        for c in DESC:
            self.meta[c] = desc[c]

    @torch.no_grad()
    def decode(self, z):
        """z: (b, L) latent vectors (numpy or tensor) -> (b, 158, 158) matrices in nm."""
        z = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=self.dev)
        h = self.model.from_z(z).view(-1, 128, 10, 10)
        out = self.model.decoder(h)[:, :, 1:-1, 1:-1]
        out = 0.5 * (out + out.transpose(2, 3))
        return (out[:, 0] * self.scale).cpu().numpy()

    # ---------- basic statistics ----------
    def _pca(self):
        if not hasattr(self, "_pc"):
            Zc = (self.Z - self.Z.mean(0)).astype(np.float64)
            w, V = np.linalg.eigh(Zc.T @ Zc / (len(Zc) - 1))
            idx = np.argsort(w)[::-1]
            self._pc = (w[idx], V[:, idx])
        return self._pc

    def stats(self):
        """Per-dimension table and a summary of how many dimensions are really used."""
        std = self.Z.std(0)
        w, _ = self._pca()
        cum = np.cumsum(w) / w.sum()
        table = pd.DataFrame({"dim": np.arange(self.L), "mean": self.Z.mean(0), "std": std,
                              "var_share": std ** 2 / (std ** 2).sum(), "dead": std < 0.01 * std.max()})
        summary = {"frames": len(self.Z),
                   "participation_ratio": float(w.sum() ** 2 / (w ** 2).sum()),
                   "pcs_for_90%": int(np.searchsorted(cum, 0.90) + 1),
                   "pcs_for_95%": int(np.searchsorted(cum, 0.95) + 1),
                   "pcs_for_99%": int(np.searchsorted(cum, 0.99) + 1),
                   "dead_dims": int(table["dead"].sum())}
        print(summary.to_string())
        return table

    def plot_stats(self, save=None, show=True):
        std = self.Z.std(0)
        w, _ = self._pca()
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
        axs[0].bar(np.arange(self.L), std)
        axs[0].set(xlabel="latent dimension", ylabel="std over frames", title="spread of each dimension")
        axs[1].plot(np.arange(1, self.L + 1), np.cumsum(w) / w.sum(), "o-")
        axs[1].set(xlabel="principal components of the code", ylabel="cumulative variance", ylim=(0, 1.02),
                   title="how many directions carry the variance")
        C = np.corrcoef(self.Z.T)
        im = axs[2].imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=axs[2])
        axs[2].set(title="correlation between dimensions", xlabel="dim", ylabel="dim")
        _finish(fig, save, show)

    # ---------- what do the dimensions encode ----------
    def _key(self, name):
        m = self.meta
        return {"run": m["run"], "cond": m["lay"].astype(str) + "_" + m["top"].astype(str), "seq": m["seq"],
                "group": m["group"], "tbin": m["t_ns"] // 10, "split": m["split"]}[name]

    def variance_decomposition(self, by=("run", "cond", "seq", "group", "tbin")):
        """Share of each dimension's variance explained by a grouping (eta squared, 0 to 1).
        run: high = the dimension identifies the run. cond: lay+top. tbin: 10 ns time bins (drift)."""
        Z = self.Z.astype(np.float64)
        mu = Z.mean(0)
        tot = ((Z - mu) ** 2).sum(0)
        Zdf = pd.DataFrame(Z)
        out = {}
        for name in by:
            g = Zdf.groupby(self._key(name).values)
            n = g.size().values[:, None]
            out[name] = (n * (g.mean().values - mu) ** 2).sum(0) / tot
        res = pd.DataFrame(out, index=[f"z{i}" for i in range(self.L)])
        res.loc["overall"] = pd.Series({k: float((v * tot).sum() / tot.sum()) for k, v in out.items()})
        return res

    def correlations(self, targets=None):
        """Pearson correlation of each latent dimension with quantities of the frame."""
        if targets is None:
            targets = [c for c in DESC + ["t_ns", "err_nm", "tte_ns"] if c in self.meta]
        Zdf = pd.DataFrame(self.Z, columns=[f"z{i}" for i in range(self.L)])
        return pd.DataFrame({t: Zdf.corrwith(self.meta[t]) for t in targets})

    def plot_correlations(self, targets=None, save=None, show=True):
        c = self.correlations(targets)
        fig, ax = plt.subplots(figsize=(1.2 * len(c.columns) + 3, 0.28 * self.L + 2))
        im = ax.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(c.columns)), c.columns, rotation=45, ha="right")
        ax.set_yticks(range(self.L), [f"z{i}" for i in range(self.L)])
        fig.colorbar(im, label="Pearson r")
        _finish(fig, save, show)

    def probe(self, targets=None, alpha=1.0):
        """Ridge regression from the code to a quantity. Fitted on training sequences, scored on
        held-out sequences. R2 near 1 means the code keeps that quantity (linearly)."""
        if targets is None:
            targets = [c for c in DESC + ["t_ns", "tte_ns"] if c in self.meta]
        tr = (self.meta["split"] == "train").values
        mu, sd = self.Z[tr].mean(0), self.Z[tr].std(0) + 1e-8
        Zs = ((self.Z - mu) / sd).astype(np.float64)
        rows = []
        for t in targets:
            y = self.meta[t].values.astype(np.float64)
            ok = np.isfinite(y)
            a, b = tr & ok, ~tr & ok
            if a.sum() < 10 or b.sum() < 10:
                continue
            ym = y[a].mean()
            w = np.linalg.solve(Zs[a].T @ Zs[a] + alpha * np.eye(self.L), Zs[a].T @ (y[a] - ym))

            def r2(m):
                p = Zs[m] @ w + ym
                return 1 - ((y[m] - p) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
            rows.append(dict(target=t, R2_train=r2(a), R2_heldout=r2(b), n_train=int(a.sum()), n_heldout=int(b.sum())))
        return pd.DataFrame(rows).set_index("target")

    # ---------- 2D views ----------
    def embed(self, method="pca", n=30000, seed=0, **kw):
        """2D coordinates of a random subset of frames. method: 'pca', 'umap' (needs umap-learn), 'tsne'."""
        key = (method, n, seed, tuple(sorted(kw.items())))
        if key in self._emb:
            return self._emb[key]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self.Z), min(n, len(self.Z)), replace=False))
        Z = self.Z[idx]
        zs = (Z - self.Z.mean(0)) / (self.Z.std(0) + 1e-8)
        if method == "pca":
            xy = (Z - self.Z.mean(0)) @ self._pca()[1][:, :2]
        elif method == "umap":
            import umap
            xy = umap.UMAP(n_components=2, random_state=seed, **kw).fit_transform(zs)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, random_state=seed, init="pca", **kw).fit_transform(zs)
        else:
            raise ValueError("method must be 'pca', 'umap' or 'tsne'")
        out = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "idx": idx})
        self._emb[key] = out
        return out

    def plot_embedding(self, color="group", method="pca", n=30000, categorical=None, s=3, alpha=0.5,
                       max_legend=12, save=None, show=True, **kw):
        """Scatter of the code in 2D, coloured by any column of self.meta
        (group, split, lay, top, seq, t_ns, rg, mean_d, err_nm, tte_ns, ...)."""
        emb = self.embed(method, n, **kw)
        col = self.meta[color].iloc[emb["idx"].values].reset_index(drop=True)
        cat = categorical if categorical is not None else (
            not pd.api.types.is_numeric_dtype(col) or col.nunique() <= 12)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        if cat:
            cats = ["train", "holdout"] if color == "split" else sorted(col.dropna().unique(), key=str)
            cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            for i, c in enumerate(cats):
                msk = (col == c).values
                ax.scatter(emb["x"][msk], emb["y"][msk], s=s, alpha=alpha, color=cmap(i % cmap.N),
                           label=str(c) if len(cats) <= max_legend else None)
            if len(cats) <= max_legend:
                ax.legend(markerscale=4, title=color)
        else:
            sc = ax.scatter(emb["x"], emb["y"], c=col, s=s, alpha=alpha, cmap="viridis")
            fig.colorbar(sc, label=color)
        lab = "PC" if method == "pca" else method
        ax.set(xlabel=f"{lab} 1", ylabel=f"{lab} 2", title=f"latent {self.L}: {method}, coloured by {color}")
        _finish(fig, save, show)

    # ---------- what a direction in the code means ----------
    def traverse(self, kind="pc", index=0, ks=(-3, -2, -1, 0, 1, 2, 3)):
        """Decode points along one direction through the mean code.
        kind 'pc': principal direction of the code (index 0 = largest variance); kind 'dim': one latent dimension.
        Steps are k standard deviations. Returns ks, matrices (nm) and the radius of gyration of each."""
        mu = self.Z.mean(0)
        if kind == "pc":
            w, V = self._pca()
            direction, sd = V[:, index], np.sqrt(w[index])
        else:
            direction, sd = np.eye(self.L)[index], self.Z[:, index].std()
        zs = mu[None] + np.array(ks)[:, None] * sd * direction[None]
        mats = self.decode(zs)
        iu = np.triu_indices(N, 1)
        rg = np.sqrt((mats[:, iu[0], iu[1]] ** 2).sum(1)) / N
        return ks, mats, rg

    def plot_traverse(self, kind="pc", index=0, ks=(-3, -2, -1, 0, 1, 2, 3), save=None, show=True):
        ks, mats, rg = self.traverse(kind, index, ks)
        c = list(ks).index(0) if 0 in ks else len(ks) // 2
        diff = mats - mats[c]
        vmax = np.abs(diff).max() + 1e-12
        fig, axs = plt.subplots(2, len(ks), figsize=(2.3 * len(ks), 5.2), squeeze=False)
        for j, k in enumerate(ks):
            im0 = axs[0, j].imshow(mats[j], vmin=mats.min(), vmax=mats.max(), cmap="viridis")
            im1 = axs[1, j].imshow(diff[j], vmin=-vmax, vmax=vmax, cmap="coolwarm")
            axs[0, j].set_title(f"{k:+g} sd\nRg {rg[j]:.3f} nm", fontsize=9)
            for a in axs[:, j]:
                a.axis("off")
        fig.colorbar(im0, ax=axs[0].tolist(), shrink=0.8, label="distance (nm)")
        fig.colorbar(im1, ax=axs[1].tolist(), shrink=0.8, label="change vs centre (nm)")
        fig.suptitle(f"decoded matrices along {kind} {index} of latent {self.L}")
        if save:
            fig.savefig(save, dpi=150)
        if show:
            plt.show()

    # ---------- single runs over time ----------
    def pick_runs(self, n=6, group=None, min_frames=20, seed=0):
        sizes = self.meta.groupby("run").size()
        grp = self.meta.groupby("run")["group"].first()
        ok = sizes[sizes >= min_frames].index
        rng = np.random.default_rng(seed)
        if group is not None:
            pool = [r for r in ok if grp[r] == group]
            return list(rng.choice(pool, min(n, len(pool)), replace=False))
        out = []
        for g in ["A", "B"]:
            pool = [r for r in ok if grp[r] == g]
            out += list(rng.choice(pool, min(n // 2, len(pool)), replace=False))
        return out

    def _rows(self, run):
        rows = np.flatnonzero(self.meta["run"].values == run)
        return rows[np.argsort(self.meta["pos"].values[rows])]

    def plot_trajectories(self, runs=None, n=6, save=None, show=True):
        """Paths of runs through the first two principal directions of the code.
        Circle = start, square = end of a run that never unfolds, X = end of a run that unfolds (group B)."""
        runs = runs or self.pick_runs(n)
        _, V = self._pca()
        mu = self.Z.mean(0)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for i, r in enumerate(runs):
            rows = self._rows(r)
            xy = (self.Z[rows] - mu) @ V[:, :2]
            g = self.meta["group"].values[rows[0]]
            ax.plot(xy[:, 0], xy[:, 1], "-", lw=0.8, color=f"C{i % 10}", alpha=0.6, label=f"{r} ({g})")
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=self.meta["t_ns"].values[rows], s=8, cmap="viridis")
            ax.plot(xy[0, 0], xy[0, 1], "o", color=f"C{i % 10}", mec="k", ms=8)
            ax.plot(xy[-1, 0], xy[-1, 1], "X" if g == "B" else "s", color=f"C{i % 10}", mec="k", ms=9)
        fig.colorbar(sc, label="time (ns)")
        ax.set(xlabel="PC 1", ylabel="PC 2", title=f"runs in latent space (latent {self.L})")
        ax.legend(fontsize=7)
        _finish(fig, save, show)

    def plot_run_timeseries(self, run=None, dims=None, n_dims=6, save=None, show=True):
        """Latent dimensions of one run against time, with its Rg and reconstruction error."""
        run = run or self.pick_runs(1, group="B")[0]
        rows = self._rows(run)
        t = self.meta["t_ns"].values[rows]
        dims = list(dims) if dims is not None else list(np.argsort(-self.Z.std(0))[:n_dims])
        fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for d in dims:
            axs[0].plot(t, self.Z[rows, d], label=f"z{d}")
        axs[0].set(ylabel="latent value", title=f"{run} ({self.meta['group'].values[rows[0]]})")
        axs[0].legend(ncol=3, fontsize=8)
        axs[1].plot(t, self.meta["rg"].values[rows], color="k")
        axs[1].set(ylabel="Rg (nm)")
        if "err_nm" in self.meta:
            axs[2].plot(t, self.meta["err_nm"].values[rows], color="tab:red")
        axs[2].set(ylabel="reconstruction RMSE (nm)", xlabel="time (ns)")
        _finish(fig, save, show)

    # ---------- coverage of held-out sequences ----------
    def novelty(self, n_query=3000, n_ref=50000, seed=0):
        """Distance in code space to the nearest training frame from a DIFFERENT sequence.
        Held-out frames are compared with training frames that are treated the same way."""
        rng = np.random.default_rng(seed)
        split = self.meta["split"].values
        seq = pd.factorize(self.meta["seq"])[0]
        tr = np.flatnonzero(split == "train")
        ho = np.flatnonzero(split == "holdout")
        ref = rng.choice(tr, min(n_ref, len(tr)), replace=False)
        sets = {"held-out frames": rng.choice(ho, min(n_query, len(ho)), replace=False),
                "training frames (other sequences)": rng.choice(tr, min(n_query, len(tr)), replace=False)}
        Zr = torch.tensor(self.Z[ref], device=self.dev)
        sr = torch.tensor(seq[ref], device=self.dev)
        out = []
        for name, q in sets.items():
            Zq = torch.tensor(self.Z[q], device=self.dev)
            sq = torch.tensor(seq[q], device=self.dev)
            for a in range(0, len(q), 1000):
                d = torch.cdist(Zq[a:a + 1000], Zr)
                d[sq[a:a + 1000, None] == sr[None, :]] = float("inf")
                out.append(pd.DataFrame({"kind": name, "nn_dist": d.min(1).values.cpu().numpy(),
                                         "row": q[a:a + 1000]}))
        return pd.concat(out, ignore_index=True)

    def plot_novelty(self, save=None, show=True, **kw):
        d = self.novelty(**kw)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.logspace(np.log10(max(d["nn_dist"].min(), 1e-4)), np.log10(d["nn_dist"].max()), 80)
        for k, g in d.groupby("kind"):
            ax.hist(g["nn_dist"], bins=bins, density=True, alpha=0.5, label=f"{k} (median {g['nn_dist'].median():.3f})")
        ax.set(xscale="log", xlabel="distance to nearest training frame of another sequence (code units)",
               ylabel="density", title="coverage of held-out sequences in latent space")
        ax.legend(fontsize=8)
        _finish(fig, save, show)
        return d.groupby("kind")["nn_dist"].describe()