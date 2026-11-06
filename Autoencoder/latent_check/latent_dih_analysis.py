
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



# ---------------------------------------------------------------------
# model <-> arrays
# ---------------------------------------------------------------------
def encode(model, x, device="cpu", batch=16384):
    """(N, 2D) model input -> (N, latent) latent codes."""
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


def embed_2d(z, method="pca", max_points=20000, seed=0):
    """-> (xy (M, 2), selected indices (M,)). 't-SNE' and 'umap' need scikit-learn / umap-learn."""
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(z), min(max_points, len(z)), replace=False))
    zs = z[sel]
    if method == "pca":
        p = pca(zs)
        return p["scores"][:, :2], sel
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(zs), sel
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(zs), sel
    raise ValueError("method must be 'pca', 'tsne' or 'umap'")


def scatter_maps(xy, panels, size=4):
    """One scatter per entry of `panels` (name -> values for the same points). Text/object values are treated as
    categories, numbers as a colour scale."""
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 4), squeeze=False)
    for ax, (name, v) in zip(axs[0], panels.items()):
        v = np.asarray(v)
        if v.dtype == object or v.dtype.kind in "US":
            codes = np.unique(v, return_inverse=True)[1]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=codes, s=size, cmap="tab20" if codes.max() < 20 else "nipy_spectral")
            ax.set_title(f"{name} ({codes.max() + 1} groups)")
        else:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=size, cmap="viridis")
            ax.set_title(name); fig.colorbar(sc, ax=ax)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); plt.show()


def latent_dihedral_corr(scores, true_deg, valid, names, n_dims=16, top=3):
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