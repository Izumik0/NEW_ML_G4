import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ---------- DATA ------------------
CSV = '../lifetimes_data_simple.csv'
NPZ_DIR = Path('../distance_matrix-kopia/')
Cache_dir = Path('cache'); Cache_dir.mkdir(exist_ok=True)
NPZ_key = None
Store_DTYPE = np.float16
Include_C=True


# ---------- CONFIG --------------
DT_PS = 100
Stride = 10 # take every 1ns from data and puts it input cache and training
N = 158 # size of matrix (N x N)
Latents = [2,4,8,16,32,64] # bottelneck of encoder
MAP_Latent= 16
Holdout = 0.2 # 20% of cache data is not seen in training by model (generalizaion check)
Max_train_frame = 100000 #how much frames is used in training
Epoches = 20
Batch = 32
LR = 1e-3
WD = 1e-4
PCA_frames=10000
EVAL_subset = 1 #it determinates which frames are taken to testing (1 means all frames)
BIN_ns=10
SEED=42
Out=Path("../Autoencoder_results_v1"); Out.mkdir(exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
IU = np.triu_indices(N,k=1)
IU_T = torch.triu_indices(N,N,offset=1)

df = pd.read_csv(CSV, index_col=0)
df["group"]=np.where(df.unf_time >=100_000, "A", np.where(df.unf_time >0, "B", "C"))

def parse_name(p):
    parts = p.stem[len("distance_arr_"):].split("_")
    return dict(seq=int(parts[0]), top="_".join(parts[1:-2]), lay=parts[-2], copy=int(parts[-1]), npz=p)

found = pd.DataFrame([parse_name(p) for p in sorted(NPZ_DIR.glob("distance_arr_*.npz"))])
KEYS = ["seq", "top", "lay", "copy"]
print(f"{len(found)} .npz files found in {NPZ_DIR}")
orphans = found.merge(df[KEYS], on=KEYS, how="left", indicator=True)
print("files with no matching CSV row:", int((orphans._merge == "left_only").sum()))
df = df.merge(found, on=KEYS, how="left")
df["has_npz"] = df.npz.notna()
print(df.groupby("group").has_npz.agg(has_file="sum", rows="size"))
df = df[df.has_npz & ((df.group != "C") | Include_C)].reset_index(drop=True)
df["npy"] = [Cache_dir / f"{p.stem}.npy" for p in df.npz]

seqs = df.seq.unique()
held = set(rng.choice(seqs, max(1, round(len(seqs) * Holdout)), replace=False))
df['split']=np.where(df.seq.isin(held), 'holdout', 'train')

# Cache work
def ensure_cache(r):
    p=r.npy
    if not p.exists():
        with np.load(r.npz) as z:
            key = NPZ_key or (z.files[0] if len(z.files)==1 else None)
            if key is None:
                raise KeyError(f"{r.npz.name} has several arrayes {z.files}; set NPZ_Key")
            a = z[key]
        if a.shape[1:] != (N,N):  # checking shape
            if a.shape[2:] == (N,N):
                a = np.moveaxis(a, -1,0)
            else:
                raise ValueError(f"{r.npz.name} has unexpected shape: {a.shape}")
        tmp = p.with_suffix('.tmp.npy')
        np.save(tmp, a[::Stride][:, IU[0], IU[1]].astype(Store_DTYPE))
        tmp.rename(p)
    return np.load(p, mmap_mode='r').shape[0]

ms = []
for i, r in enumerate(df.itertuples()):
    ms.append(ensure_cache(r))
    if i%5 == 0:
        print(f"cached {i}/{len(df)} runs")
df['m']=ms
df=df[df.m>0].reset_index(drop=True)

# Data magic work
class Frames(Dataset):
    def __init__(self, paths, items, scale):
        self.paths, self.items, self.scale = paths, items, scale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        k, pos = self.items[idx]
        v = np.asarray(np.load(self.paths[k], mmap_mode='r')[pos], dtype=np.float32) / self.scale
        x = np.zeros((N,N), np.float32)
        x[IU]=v
        return torch.from_numpy(x + x.T).unsqueeze(0), k, pos

def all_it(runs):
    return [(k,p) for k,m in enumerate(runs.m.values) for p in range(m)]

paths = list(df.npy)
train_rows = np.flatnonzero((df.split == "train").values)
train_items = [(int(k), p) for k in train_rows for p in range(df.m[k])]
sel = rng.choice(len(train_items), min(Max_train_frame, len(train_items)), replace=False)
train_items = [train_items[i] for i in sel]
eval_items = all_it(df)[::EVAL_subset]
print(f"training frames: {len(train_items)}, test frames: {len(eval_items)}")

# scale and total variance from a training sample (train data only)
samp = [train_items[i] for i in rng.choice(len(train_items), min(PCA_frames, len(train_items)), replace=False)]
V0 = np.stack([np.asarray(np.load(paths[k], mmap_mode="r")[p], dtype=np.float32) for k, p in samp])
SCALE = float(np.percentile(V0, 99.9))
V0 = torch.from_numpy(V0 / SCALE)
MU = V0.mean(0)
VAR_TOTAL = float(((V0 - MU) ** 2).mean())  # variance around the mean structure (scaled units)
print(f"scale {SCALE:.2f} A; predicting the mean structure gives RMSE {np.sqrt(VAR_TOTAL) * SCALE:.3f} A")

# ---------------- MODELS ---------------------

def blok(i,o):
    x = nn.Sequential(nn.Conv2d(i,o,3, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU())
    return x

def up(i,o):
    x = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.Conv2d(i,o,3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
    return x

class ConvEncoder(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.encoder = nn.Sequential(blok(1,16), blok(16,32),blok(32,64),blok(64, 128))
        self.to_z= nn.Linear(128*10*10, latent)
        self.from_z = nn.Linear(latent, 128*10*10)
        self.decoder = nn.Sequential(up(128,64),up(64,32),up(32,16),
                                     nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(16,1,3, padding=1))

    def forward(self, x):
        x=F.pad(x, (1,1,1,1), mode='replicate')
        z=self.to_z(self.encoder(x).flatten(1))
        out = self.decoder(self.from_z(z).view(-1,128,10,10))[:,:,1:-1,1:-1]
        return 0.5*(out+out.transpose(2,3)), z

def recon_loss(out,x):
    d=(out-x)[:,0,IU_T[0],IU_T[1]]
    return (d**2).mean(dim=1)

def train(latent):
    model = ConvEncoder(latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoches)
    dl = DataLoader(Frames(paths, train_items, SCALE), batch_size=Batch, shuffle=True, num_workers=0)
    for ep in range(Epoches):
        model.train(); total=n=0
        for x, _, _ in dl:
            x=x.to(device)
            loss=recon_loss(model(x)[0], x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()*len(x); n+=len(x)

        scheduler.step()
        print(f" L={latent} | epoch: {ep:2d} | Train mse: {total/n:.5f}")
    return model.eval()

# PCA baselinfe for comparing neural network encoder and linear compression
_,_,PCA_V = torch.pca_lowrank(V0-MU, q=max(Latents), center=False, niter=4)

@torch.no_grad()
def eval(model, latent, with_pca=True,map_rows=None):

    enc, pca, zs = [], {L:[] for L in Latents}, []
    emap, nmap = torch.zeros(N,N,device=device), 0
    held_rows = torch.from_numpy((df.split=="holdout").values)
    for x, k , p in DataLoader(Frames(paths, eval_items, SCALE), batch_size=Batch, num_workers=0):
        xd=x.to(device)
        out=model(xd)[0]
        enc.append(recon_loss(out, xd).cpu())

        if with_pca:
            xc = x[:, 0, IU_T[0], IU_T[1]] - MU
            z = xc@PCA_V
            for L in Latents:
                pca[L].append(((xc - z[:, :L] @ PCA_V[:, :L].T) ** 2).mean(1))

        if map_rows:
            m = held_rows[k].to(device)
            if m.any():
                emap += ((out - xd) ** 2)[m, 0].sum(0)
                nmap += int(m.sum())
    res = {'enc':torch.cat(enc).numpy()}
    if with_pca:
        res["pca"] = {L: torch.cat(v).numpy() for L, v in pca.items()}
    if map_rows:
        res["map"] = (emap / max(nmap, 1)).cpu().numpy()
    return res


meta = pd.DataFrame(eval_items, columns=["k", "pos"]).join(df[["seq", "top", "lay", "copy", "group", "split"]], on="k")
meta["t_ns"] = meta.pos * Stride * DT_PS / 1000
for i, L in enumerate(Latents):
    print(f"\n=== bottleneck {L}")
    model = train(L)
    r = eval(model, L, with_pca=(i == 0), map_rows=(L == MAP_Latent))
    meta[f"enc_{L}"] = r["enc"]
    if i == 0:
        for LL, e in r["pca"].items():
            meta[f"pca_{LL}"] = e
    if L == MAP_Latent:
        np.save(Out / "error_map.npy", r["map"])
    torch.save({"model": model.state_dict(), "scale": SCALE}, Out / f"ae_L{L}.pt")
meta.to_csv(Out / "frame_errors.csv.gz", index=False)

# %% summary table
rows = []
for L in Latents:
    for method in ["enc", "pca"]:
        for name, d in [("all", meta), ("train seqs", meta[meta.split == "train"]),
                        ("held-out seqs", meta[meta.split == "holdout"])]:
            mse = d[f"{method}_{L}"].mean()
            rows.append(dict(latent=L, method=method, subset=name, rmse_A=np.sqrt(mse) * SCALE,
                             R2=1 - mse / VAR_TOTAL))
summary = pd.DataFrame(rows)
summary.to_csv(Out / "summary.csv", index=False)
print(summary.pivot_table(index="latent", columns=["method", "subset"], values="rmse_A").round(3))
print(f"\nreference: mean structure only -> RMSE {np.sqrt(VAR_TOTAL) * SCALE:.3f} A")

# %% plots
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for method, ls in [("enc", "-"), ("pca", "--")]:
    for name, c in [("all", "k"), ("held-out seqs", "tab:red")]:
        d = summary[(summary.method == method) & (summary.subset == name)]
        axs[0].plot(d.latent, d.rmse_A, ls, color=c, marker="o", label=f"{method.upper()}, {name}")
        axs[1].plot(d.latent, d.R2, ls, color=c, marker="o")
axs[0].set(xscale="log", yscale="log", xlabel="bottleneck size", ylabel="reconstruction RMSE (A)")
axs[0].legend();
axs[1].set(xscale="log", xlabel="bottleneck size", ylabel="variance explained (R$^2$)")
fig.tight_layout();
fig.savefig(Out / "error_vs_bottleneck.png", dpi=150)

col = f"enc_{MAP_Latent}"
meta["bin"] = (meta.t_ns // BIN_ns) * BIN_ns + BIN_ns / 2
fig, ax = plt.subplots(figsize=(6, 4))
for g, d in meta.groupby("group"):
    m = d.groupby("bin")[col].mean()
    ax.plot(m.index, np.sqrt(m) * SCALE, marker="o", label=f"group {g}")
ax.set(xlabel="time (ns)", ylabel="RMSE (A)", title=f"AE reconstruction error over time (bottleneck {MAP_Latent})")
ax.legend()
fig.tight_layout()
fig.savefig(Out / "error_vs_time.png", dpi=150)

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(np.sqrt(np.load(Out / "error_map.npy")) * SCALE, cmap="magma")
ax.set_title(f"RMSE per residue pair (A), held-out seqs, bottleneck {MAP_Latent}")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(Out / "error_map.png", dpi=150)

# error by condition: is information lost more for some topologies / layouts?
by = meta.groupby(["lay", "top"])[col].mean().pipe(lambda s: np.sqrt(s) * SCALE).rename("rmse_A")
by.sort_values().to_csv(Out / "rmse_by_condition.csv")
plt.show()


