import re
from itertools import islice
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


#some things
FILE_KEYS = ("seq", "top", "lay", "copy", "run")     # filters on these are applied before reading any file



# ---------------------------------------------------------------------
# model <-> arrays
# ---------------------------------------------------------------------
def encode(model, x, device="cpu", batch=16384):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch], dtype=np.float32)).to(device)
            out.append(model.enc(xb).cpu().numpy())
    return np.concatenate(out)


def decode_raw(model, z, device="cpu", batch=16384):
    """(N, latent) -> (N, 2D) decoder output [cos | sin]."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(z), batch):
            zb = torch.from_numpy(np.ascontiguousarray(z[i:i + batch], dtype=np.float32)).to(device)
            out.append(model.dec(zb).cpu().numpy())
    return np.concatenate(out)


def to_degrees(rec):
    D = rec.shape[1] // 2
    return np.degrees(np.arctan2(rec[:, D:], rec[:, :D]))


def decode_to_degrees(model, z, device="cpu"):
    return to_degrees(decode_raw(model, z, device))


def collect_latents(model, all_names, files, prepare_input, stride=50, max_per_run=1000, device="cpu"):
    Z, T, R, ND, E, TR, V = [], [], [], [], [], [], []
    for f in files:
        with open(f) as fh:
            names = fh.readline().lstrip("#").split()[1:]
            arr = np.loadtxt(islice(fh, 0, None, stride), dtype=np.float32, ndmin=2)
        t_ps, deg = arr[:, 0], arr[:, 1:]
        if len(deg) > max_per_run:
            sel = np.linspace(0, len(deg) - 1, max_per_run).astype(int)
            t_ps, deg = t_ps[sel], deg[sel]
        x, mask, full, valid = prepare_input(deg, names, all_names)
        valid = np.asarray(valid)
        if valid.ndim == 1:                                       # per-run mask -> per-frame mask
            valid = np.repeat(valid[None, :], len(x), 0)
        z = encode(model, x, device)
        pred = to_degrees(decode_raw(model, z, device) * mask)
        err = np.abs((pred - full + 180.0) % 360.0 - 180.0)
        err[~valid] = np.nan
        Z.append(z); T.append(t_ps / 1000.0); R.append(np.full(len(z), Path(f).parent.name, dtype=object))
        ND.append(valid.sum(1)); E.append(np.nanmean(err, 1))
        TR.append(np.where(valid, full, np.nan)); V.append(valid)
    return dict(z=np.concatenate(Z), t_ns=np.concatenate(T), run=np.concatenate(R), n_dih=np.concatenate(ND),
                err=np.concatenate(E), true=np.concatenate(TR), valid=np.concatenate(V))


# ---------------------------------------------------------------------
# run labels (seq / top / lay / copy) and filters
# ---------------------------------------------------------------------
_RUN_RE = re.compile(r"^(\d+)_(.+)_([^_]+)_(\d+)$")


def parse_run_label(name):
    m = _RUN_RE.match(str(name))
    if not m:
        return dict(seq="?", top="?", lay="?", copy="?")
    seq, top, lay, copy = m.groups()
    return dict(seq=seq, top=top, lay=lay, copy=copy)


def add_labels(data, parser=parse_run_label, extra=None):
    runs = np.asarray(data["run"]).astype(str)
    uniq, inv = np.unique(runs, return_inverse=True)
    parsed = [parser(u) for u in uniq]
    for k in parsed[0]:
        data[k] = np.array([pp[k] for pp in parsed], dtype=object)[inv]
    if extra is not None:
        for c in extra.columns:
            data[c] = extra[c].reindex(uniq).to_numpy()[inv]
    if (data.get("seq") == "?").all():
        print("WARNING: no run name matched <seq>_<top>_<lay>_<copy>; pass parser=... to add_labels")
    return data


def describe_labels(data, keys=("seq", "top", "lay", "copy")):
    runs = np.asarray(data["run"]).astype(str)
    for k in keys:
        if k not in data:
            continue
        vals = np.asarray(data[k]).astype(str)
        u, c = np.unique(vals, return_counts=True)
        nr = [len(set(runs[vals == x])) for x in u]
        print(f"{k}: " + ", ".join(f"{x} ({r} runs, {n:,} frames)" for x, r, n in zip(u, nr, c)))


def _match(v, cond):
    v = np.asarray(v)
    if callable(cond):
        if v.dtype.kind in "fiub":
            return np.asarray(cond(v), bool)
        u, inv = np.unique(v.astype(str), return_inverse=True)
        return np.array([bool(cond(x)) for x in u])[inv]
    if v.dtype.kind in "fiub":                                    # numeric field
        if isinstance(cond, tuple) and len(cond) == 2:            # (min, max); None = open end
            lo, hi = cond
            ok = np.isfinite(v)
            if lo is not None:
                ok &= v >= lo
            if hi is not None:
                ok &= v <= hi
            return ok
        conds = [cond] if np.isscalar(cond) else list(cond)
        return np.isin(v, conds)
    vs = v.astype(str)                                            # categorical field
    conds = [str(cond)] if np.isscalar(cond) else [str(c) for c in cond]
    inc = [c for c in conds if not c.startswith("!")]
    exc = [c[1:] for c in conds if c.startswith("!")]
    ok = np.isin(vs, inc) if inc else np.ones(len(vs), bool)
    if exc:
        ok &= ~np.isin(vs, exc)
    return ok


match_values = _match


def make_filter(data, **filters):

    n = len(data["z"])
    keep = np.ones(n, bool)
    skip = {"z", "true", "valid"}
    for key, cond in filters.items():
        if key not in data or key in skip:
            raise KeyError(f"unknown filter '{key}'. Available: {sorted(k for k in data if k not in skip)}")
        keep &= _match(data[key], cond)
    nr = len(set(np.asarray(data["run"])[keep].astype(str)))
    print(f"filter keeps {int(keep.sum()):,} of {n:,} frames ({nr} runs)")
    return keep


# ---------------------------------------------------------------------
# structure of the latent space
# ---------------------------------------------------------------------
def pca(z):
    """PCA of the latent codes. The AE's own axes have no order or meaning; PCA axes are ordered by variance."""
    mu = z.mean(0)
    zc = z - mu
    _, S, Vt = np.linalg.svd(zc, full_matrices=False)
    var = S ** 2 / max(len(z) - 1, 1)
    return dict(mean=mu, Vt=Vt, var=var, ratio=var / var.sum(), scores=zc @ Vt.T,
                pr=float(var.sum() ** 2 / (var ** 2).sum()))


def plot_overview(z, p):
    std = z.std(0)
    active = int((std > 0.01 * std.max()).sum())
    cum = np.cumsum(p["ratio"])
    print(f"latent size {z.shape[1]}: {active} dimensions with std > 1% of the largest one")
    print(f"PCA: {int(np.searchsorted(cum, 0.90)) + 1} components for 90% of the variance, "
          f"{int(np.searchsorted(cum, 0.99)) + 1} for 99%; participation ratio {p['pr']:.1f}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(range(len(std)), std, color="tab:blue")
    ax[0].set(xlabel="latent dimension", ylabel="std of the code", title="Spread per dimension (near 0 = unused)")
    ax[1].plot(range(1, len(cum) + 1), cum, "o-", ms=3)
    ax[1].axhline(0.9, color="gray", ls=":"); ax[1].axhline(0.99, color="gray", ls=":")
    ax[1].set(xlabel="PCA components", ylabel="cumulative variance", title="Effective dimensionality")
    keep = std > 0.01 * std.max()
    c = np.corrcoef(z[:, keep].T) if keep.sum() > 1 else np.ones((1, 1))
    im = ax[2].imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax[2].set_title("Correlation between active dimensions"); fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); plt.show()


def embed_2d(z, method="pca", max_points=20000, seed=0, mask=None, pca_fit="all"):
    """
    -> (xy (M, 2), sel (M,)): coordinates and the indices (into the arrays of `data`) of the frames shown.
    mask    : boolean filter from make_filter(); only these frames are shown (None = all frames)
    pca_fit : 'all'      = the PCA axes are fitted on ALL frames, so different filters share the same axes
                           and are directly comparable;
              'selected' = the axes are fitted on the filtered frames only.
              t-SNE and UMAP are always fitted on the filtered frames, so their layout changes with the filter.
    """
    rng = np.random.default_rng(seed)
    cand = np.arange(len(z)) if mask is None else np.flatnonzero(mask)
    if len(cand) == 0:
        raise ValueError("the filter removed every frame")
    sel = np.sort(rng.choice(cand, min(max_points, len(cand)), replace=False))
    if method == "pca":
        fit = z if pca_fit == "all" else z[cand]
        if len(fit) > 200000:
            fit = fit[rng.choice(len(fit), 200000, replace=False)]
        mu = fit.mean(0)
        Vt = np.linalg.svd(fit - mu, full_matrices=False)[2]
        return (z[sel] - mu) @ Vt[:2].T, sel
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", perplexity=min(30, max(5, len(sel) // 4)),
                    random_state=seed).fit_transform(z[sel]), sel
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(z[sel]), sel
    raise ValueError("method must be 'pca', 'tsne' or 'umap'")


_PANEL_TITLES = {"t_ns": "time (ns)", "err": "frame error (deg)", "n_dih": "number of dihedrals",
                 "seq": "sequence", "top": "topology", "lay": "lay", "copy": "copy", "run": "run"}


def default_panels(data, sel, keys=("run", "seq", "top", "t_ns", "err", "n_dih")):
    """{title: values} for scatter_maps, in the given order, for the frames `sel`."""
    return {_PANEL_TITLES.get(k, k): np.asarray(data[k])[sel] for k in keys if k in data}


def _category_colors(n):
    if n <= 10:
        return plt.cm.tab10(np.arange(n))
    if n <= 20:
        return plt.cm.tab20(np.arange(n))
    return plt.cm.nipy_spectral(np.linspace(0.05, 0.95, n))


def scatter_maps(xy, panels, size=5, max_legend=20, alpha=0.85, suptitle=None, ncols=3, save=None, show=False):
    """
    One scatter per entry of `panels` (title -> values for the same points).
    Text values are categories: every category gets a colour and a legend entry with its number of points. With
    more than `max_legend` categories the largest ones are coloured and listed, the rest are grey ('other').
    Numbers get a colour bar labelled with the panel title. Panels wrap into rows of `ncols`.
    """
    n = len(panels)
    cols_n = min(ncols, n)
    rows_n = int(np.ceil(n / cols_n))
    fig, axs = plt.subplots(rows_n, cols_n, figsize=(5.6 * cols_n, 5.4 * rows_n), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, (name, v) in zip(axs.ravel(), panels.items()):
        v = np.asarray(v)
        if v.dtype == object or v.dtype.kind in "US":
            vs = v.astype(str)
            cats, inv, cnt = np.unique(vs, return_inverse=True, return_counts=True)
            order = np.argsort(cnt)[::-1]
            shown = order[:max_legend]
            cols = _category_colors(len(shown))
            rgba = np.tile(np.array([0.78, 0.78, 0.78, 1.0]), (len(vs), 1))
            handles = []
            for rank, ci in enumerate(shown):
                rgba[inv == ci] = cols[rank]
                handles.append(Line2D([], [], marker="o", ls="", color=cols[rank], markersize=6,
                                      label=f"{cats[ci]} (n={cnt[ci]:,})"))
            rest = len(cats) - len(shown)
            if rest > 0:
                handles.append(Line2D([], [], marker="o", ls="", color=(0.78, 0.78, 0.78), markersize=6,
                                      label=f"other ({rest} groups, n={cnt[order[len(shown):]].sum():,})"))
            ax.scatter(xy[:, 0], xy[:, 1], c=rgba, s=size, alpha=alpha, linewidths=0)
            ax.set_title(f"{name}: {len(cats)} groups")
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=7,
                      frameon=False, title=f"colour = {name}", title_fontsize=8)
        else:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=size, cmap="viridis", alpha=alpha, linewidths=0)
            ax.set_title(name)
            fig.colorbar(sc, ax=ax, label=name)
        ax.set_xticks([]); ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def highlight_groups(xy, values, title="group", max_groups=12, ncols=4, size=5, save=None, show=False):
    """Small multiples: one panel per category, that category in colour on top of all frames in grey.
    Shows where each group sits and how much groups overlap. Largest `max_groups` categories only."""
    v = np.asarray(values).astype(str)
    cats, cnt = np.unique(v, return_counts=True)
    order = np.argsort(cnt)[::-1][:max_groups]
    n = len(order)
    cols_n = min(ncols, n)
    rows_n = int(np.ceil(n / cols_n))
    fig, axs = plt.subplots(rows_n, cols_n, figsize=(4.2 * cols_n, 4.0 * rows_n), squeeze=False, sharex=True, sharey=True)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    colors = _category_colors(n)
    for ax, ci, c in zip(axs.ravel(), order, colors):
        ax.scatter(xy[:, 0], xy[:, 1], s=size * 0.6, color=(0.85, 0.85, 0.85), linewidths=0)
        m = v == cats[ci]
        ax.scatter(xy[m, 0], xy[m, 1], s=size, color=c, linewidths=0)
        ax.set_title(f"{cats[ci]} (n={cnt[ci]:,})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"each {title} highlighted (colour) against all frames (grey)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def latent_dihedral_corr(scores, true_deg, valid, names, n_dims=12, top=3):
    """
    How strongly each (PCA-ordered) latent axis relates to each dihedral: max(|corr with cos|, |corr with sin|),
    computed only over frames where the dihedral exists. Values near 1 = that axis tracks that dihedral.
    """
    K = min(n_dims, scores.shape[1])
    D = true_deg.shape[1]
    C = np.full((K, D), np.nan)
    r = np.deg2rad(np.nan_to_num(true_deg))
    for d in range(D):
        v = valid[:, d]
        if v.sum() < 30:
            continue
        s = scores[v, :K]
        s = (s - s.mean(0)) / (s.std(0) + 1e-12)
        for f in (np.cos(r[v, d]), np.sin(r[v, d])):
            if f.std() < 1e-6:
                continue
            c = np.abs(s.T @ ((f - f.mean()) / f.std())) / v.sum()
            C[:, d] = np.fmax(C[:, d], c)
    fig, ax = plt.subplots(figsize=(16, 0.35 * K + 2.5))
    im = ax.imshow(np.ma.masked_invalid(C), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"PC{i + 1}" for i in range(K)])
    ax.set_xticks(range(D)); ax.set_xticklabels(names, rotation=90, fontsize=5)
    fig.colorbar(im, label="|correlation|"); fig.tight_layout(); plt.show()
    na = np.array(names)
    for k in range(min(K, 5)):
        o = np.argsort(np.nan_to_num(C[k], nan=-1))[::-1][:top]
        print(f"PC{k + 1}: " + ", ".join(f"{na[j]} ({C[k, j]:.2f})" for j in o))
    return C


# ---------------------------------------------------------------------
# decoding: what do the axes and the space between frames mean?
# ---------------------------------------------------------------------
def plot_paths(ang, xvals, names, valid_cols, xlabel, title, top=8):
    """Decoded angles along a path in latent space; shows the `top` dihedrals that change most."""
    ang = np.asarray(ang, float).copy()
    ang[:, ~np.asarray(valid_cols, bool)] = np.nan
    ref = ang[len(ang) // 2]
    d = (ang - ref + 180.0) % 360.0 - 180.0
    spread = np.nanmax(d, 0) - np.nanmin(d, 0)
    order = np.argsort(np.nan_to_num(spread, nan=-1))[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j in order:
        ax.plot(xvals, np.degrees(np.unwrap(np.radians(ang[:, j]))), marker="o", ms=3, label=f"{names[j]} ({spread[j]:.0f}°)")
    ax.set(xlabel=xlabel, ylabel="decoded angle (deg, unwrapped)", title=title)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); plt.show()
    return spread


def traverse_pc(decode_fn, p, pc, names, valid_cols, span=3.0, n=9):
    """Move along one PCA axis of the latent space (from -span to +span standard deviations) and decode."""
    ts = np.linspace(-span, span, n)
    sd = np.sqrt(p["var"][pc])
    z_path = p["mean"][None, :] + ts[:, None] * sd * p["Vt"][pc][None, :]
    ang = decode_fn(z_path)
    plot_paths(ang, ts, names, valid_cols, "position along PC (std units)", f"Decoded angles along PC{pc + 1}")
    return ang


def interpolate(decode_fn, z_a, z_b, names, valid_cols, n=11):
    """Straight line between the codes of two frames, decoded. A smooth path = a well-behaved latent space."""
    ts = np.linspace(0, 1, n)
    ang = decode_fn((1 - ts)[:, None] * z_a[None, :] + ts[:, None] * z_b[None, :])
    plot_paths(ang, ts, names, valid_cols, "0 = frame A, 1 = frame B", "Decoded angles between two frames")
    return ang


def error_vs_distance(z, err):
    """Do frames far from the centre of the code cloud reconstruct worse? Prints a rank correlation."""
    dist = np.linalg.norm(z - z.mean(0), axis=1)
    ok = np.isfinite(err)
    rk = lambda a: np.argsort(np.argsort(a)).astype(float)
    rho = np.corrcoef(rk(dist[ok]), rk(err[ok]))[0, 1]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(dist[ok], err[ok], s=3, alpha=0.4)
    ax.set(xlabel="distance from the centre of the codes", ylabel="frame error (deg)", title=f"rank correlation {rho:.2f}")
    fig.tight_layout(); plt.show()
    return rho