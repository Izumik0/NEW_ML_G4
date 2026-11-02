import copy
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re, glob, os, sys
import pandas as pd
import random
import matplotlib.pyplot as plt

# -------- Config -------------

# --------- Files config --------------
IN_DIR = "../Autoencoder/cache_dihedrals"
OUT_DIR = "dihedrals_ts_test/"
csv_file = "../lifetimes_data_simple.csv"

os.makedirs(OUT_DIR, exist_ok=True)
# ---------- Data specific config ------------
N_Cols = 118 # how many dihedral angels does you have in your backbone eq.: res * 6 - 2 eg. 20*6-2 = 118
Frames_ps = 20*25 # need to match Autoencoder cache config dt * STRIDE. If you dont use Autoencoder then it is your dt from calc
H_ns = 10 # unfolding prediction horizon in nanoseconds
# ------------ model-related confiog -------------
bs = 5
drop = 0.2
hidden_dim = 128
layers = 2
Epoches = 25
LR = 1e-4
Patience = 6


# ---------- some variables that you dont change ------------
H_fr = int(round(H_ns*1000/Frames_ps))
Name_re = re.compile(r"dihedrals_(\d{3})_([^_]+)_([^_]+)_(\d+)_bb_dihedrals\.txt_s25\.npy$")

# ----------- hardware things ----------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if (device == torch.device("cuda") and sys.platform.startswith("linux")):
 n_workers = 4
else:
 n_workers = 0

print(f'device:{device}, n_workers:{n_workers}')
# ------------ helpers --------------

def load_run(path):
    a = np.load(path).astype(np.float32)
    T, n = a.shape
    a = np.nan_to_num(a)
    rad = np.deg2rad(a)
    x = np.zeros((T, 2*N_Cols), np.float32)
    x[:, :n] = np.cos(rad)
    x[:, N_Cols:N_Cols+n] = np.sin(rad)
    present=np.zeros(N_Cols, bool)
    present[:n] = True
    return x, present

def parse_names(path):
    m = Name_re.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"{path} is not a valid file")
    seq, top, lay, copy = m.groups()
    return int(seq), top.strip(), lay.strip(), int(copy)

def build_items(cache_dir, csv_path, min_unf_ps = 2*H_ns*1000):
    df=pd.read_csv(csv_path)
    df['seq']=df['seq'].astype(int)
    df['copy']=df['copy'].astype(int)
    df['top']=df['top'].astype(str).str.strip()
    df['lay']=df['lay'].astype(str).str.strip()
    key = df.set_index(["seq", "top", "lay", "copy"])["unf_time"]
    assert key.index.is_unique, "dupicated rows in csv file"

    items, unmatch, too_short = [], [], []
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.npy"))):
        k = parse_names(p)
        if k not in key.index:
            unmatch.append(p)
            continue
        u = float(key.loc[k])
        if u < 1000000 and u < min_unf_ps:
            too_short.append((os.path.basename(p),u))
            continue
        items.append((p,u))

    n_stable = sum(u >= 1000000 for _, u in items)
    print(f'{len(items)} runs mached with {n_stable} stable runs and {len(items)-n_stable} other runs')

    for p in unmatch[:5]:
        print("unmached files:", os.path.basename(p))

    for name, u in too_short[:5]:
        print("too_short:", name[-45:],u)
    return items, unmatch

def make_labels(T, unf_ps, h_fr = H_fr):
    if unf_ps is None or unf_ps >=1000000:
        return np.zeros(T, np.float32), T
    unf_fr = int(unf_ps // Frames_ps)
    T_used = min(T, unf_fr)
    y = np.zeros(T_used, np.float32)
    y[max(0, unf_fr - h_fr):] = 1.0
    return y, T_used

class UnfDB(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, unf_ps = self.items[i]
        x, present = load_run(path)
        y,T = make_labels(len(x), unf_ps)
        x = np.concatenate([x[:T], np.tile(present.astype(np.float32), (T, 1))], 1)
        return torch.from_numpy(x), torch.from_numpy(y)

def coll(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs])
    X = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    Y = nn.utils.rnn.pad_sequence(ys, batch_first=True)
    M = (torch.arange(X.shape[1])[None] < lens[:, None]).float() # time mask for data
    return X, Y, M, lens

def split(items, test_frac=0.2, seed=42):
    groups={}
    for it in items:
        groups.setdefault(parse_names(it[0])[:3], []).append(it)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = max(1, int(len(keys) * test_frac))
    ts = [it for k in keys[:n_test] for it in groups[k]]
    tr = [it for k in keys[n_test:] for it in groups[k]]
    return tr, ts

@torch.no_grad()
def get_probs(model, items):
    model.eval()
    out = []
    for it in items:
        X, _ = UnfDB([it])[0]
        logits, _ = model(X[None].to(device))
        out.append(torch.sigmoid(logits[0]).cpu().numpy())
    return out

def first_alarm(p, tau, k=3):
    if len(p) < k:
        return None
    hit = np.convolve((p > tau).astype(int), np.ones(k, int), "valid") == k
    return int(np.argmax(hit)) + k - 1 if hit.any() else None

def plot_training(hist):
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["test"], label="test")
    plt.xlabel("epoch"); plt.ylabel("weighted BCE"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR + "/training_curves.png", dpi=150)

def plot_traces(items, probs, tau=0.9, k=3, n_unf=8, n_stable=4):
    unf = [(it, p) for it, p in zip(items, probs) if it[1] < 1_000_000][:n_unf]
    stb = [(it, p) for it, p in zip(items, probs) if it[1] >= 1_000_000][:n_stable]
    sel = unf + stb
    if not sel:
        print("nothing to plot"); return
    cols = 4
    rows = (len(sel) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, ((path, unf_ps), p) in zip(axes.ravel(), sel):
        ax.axis("on")
        t = np.arange(len(p)) * Frames_ps / 1000                 # ns
        ax.plot(t, p, lw=1)
        ax.axhline(tau, color="gray", ls=":")
        alarm = first_alarm(p, tau, k)
        title = "_".join(map(str, parse_names(path)))
        if unf_ps < 1_000_000:
            u = unf_ps / 1000
            ax.axvspan(u - H_ns, u, color="red", alpha=0.15)      # "close" window
            ax.axvline(u, color="red", ls="--")
            if alarm is not None:
                ax.axvline(t[alarm], color="green")
                title += f"\nlead {u - t[alarm]:.1f} ns"
            else:
                title += "\nmissed"
        else:
            title += "\nstable" + (" (false alarm)" if alarm is not None else "")
            if alarm is not None:
                ax.axvline(t[alarm], color="green")
        ax.set_title(title, fontsize=8); ax.set_ylim(0, 1)
        ax.set_xlabel("time (ns)", fontsize=8); ax.set_ylabel("P(close)", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR + f"/traces_tau{tau}.png", dpi=150)

def plot_summary(items, probs, tau=0.9, k=3):
    taus = np.linspace(0.05, 0.99, 40)
    det, fa = [], []
    for tt in taus:
        d = n_u = f = n_s = 0
        for (path, unf_ps), p in zip(items, probs):
            a = first_alarm(p, tt, k)
            if unf_ps >= 1_000_000:
                n_s += 1; f += a is not None
            else:
                n_u += 1; d += a is not None
        det.append(d / max(n_u, 1)); fa.append(f / max(n_s, 1))
    leads = []
    for (path, unf_ps), p in zip(items, probs):
        a = first_alarm(p, tau, k)
        if unf_ps < 1_000_000 and a is not None:
            leads.append(unf_ps / 1000 - a * Frames_ps / 1000)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(taus, det, label="unfolding runs detected")
    ax[0].plot(taus, fa, label="stable runs with false alarm")
    ax[0].axvline(tau, color="gray", ls=":")
    ax[0].set_xlabel("threshold tau"); ax[0].set_ylabel("fraction of runs"); ax[0].legend()
    if leads:
        ax[1].hist(leads, bins=15)
    ax[1].axvline(H_ns, color="red", ls="--", label=f"horizon {H_ns} ns")
    ax[1].set_xlabel(f"lead time before unfolding (ns), tau={tau}")
    ax[1].set_ylabel("runs"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR + "/threshold_summary.png", dpi=150)


# ------------ Model Arch ------------
class FellingCollaps(nn.Module):
    def __init__(self, n_in=3*N_Cols, hidden = hidden_dim, layers=layers, dropout=drop):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
        self.rnn = nn.GRU(hidden, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        z, h = self.rnn(self.inp(x), h)
        return self.head(z).squeeze(-1), h

# ---------- Traning -------------
def run_epoch(model, loader, loss_fun, device=device, opt=None):
    model.train(opt is not None)
    tot, n = 0.0, 0.0
    with torch.set_grad_enabled(opt is not None):
        for X, Y, M, _ in loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            logits, _ = model(X)
            loss = (loss_fun(logits, Y)*M).sum()/M.sum()

            if opt:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item()*M.sum().item()
            n+=M.sum().item()
    return tot/n

def train(items, epochs=Epoches, batch_size=bs, learning_rate=LR, p=Patience):
    tr, ts = split(items)
    dl_tr = DataLoader(UnfDB(tr), batch_size=batch_size, shuffle=True, collate_fn=coll)
    dl_ts = DataLoader(UnfDB(ts), batch_size=batch_size, shuffle=False, collate_fn=coll)


    hist={'train': [], 'test': []}
    pos=neg=0.0
    for _, Y,M,_ in dl_tr:
        pos += (Y*M).sum().item()
        neg += ((1-Y)*M).sum().item()

        loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(neg/max(pos, 1.0),device=device))
        print(f"train runs {len(tr)}, test runs {len(ts)}, pos frames {pos:.0f}, neg frames {neg:.0f}")

    model = FellingCollaps().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best, bad = 1e9, 0
    best_so_far=None
    for ep in range(epochs):
        l_tr = run_epoch(model, dl_tr, loss_fn, device=device, opt=opt)
        l_ts = run_epoch(model, dl_ts, loss_fn, device=device)
        print(f"epoch {ep:02d}  train {l_tr:.4f}  val {l_ts:.4f}")
        hist['train'].append(l_tr); hist['test'].append(l_ts)
        if l_ts < best:
            best,bad = l_ts, 0
            best_so_far = copy.deepcopy(model.state_dict())

        else:
            bad +=1
            if bad >= p: break

    model.load_state_dict(best_so_far)
    return model, ts, hist

@torch.no_grad()
def eval_stopping(model, val_items, taus=(0.5, 0.7, 0.8, 0.9, 0.95), k=3):
    torch.save(model, OUT_DIR+"/model.pt")
    model.eval()
    probs = []
    for it in val_items:                           # one run at a time, as in real use
        X, _ = UnfDB([it])[0]
        logits, _ = model(X[None].to(device))
        probs.append(torch.sigmoid(logits[0]).cpu().numpy())

    print(f"{'tau':>5} | stable: false alarms | unfolding: missed / early(>{H_ns} ns) / on-time | median lead (ns)")
    for tau in taus:
        fa = ns = miss = early = ok = 0
        leads = []
        for (path, unf_ps), p in zip(val_items, probs):
            hit = np.convolve((p > tau).astype(int), np.ones(k, int), "valid") == k
            alarm = int(np.argmax(hit)) + k - 1 if hit.any() else None
            if unf_ps >= 1_000_000:
                ns += 1; fa += alarm is not None
            else:
                unf_fr = int(unf_ps // Frames_ps)
                if alarm is None: miss += 1; continue
                lead = (unf_fr - alarm) * Frames_ps / 1000
                leads.append(lead)
                if lead > H_ns: early += 1
                else: ok += 1
        med = f"{np.median(leads):.1f}" if leads else "-"
        print(f"{tau:5.2f} | {fa}/{ns} | {miss} / {early} / {ok} | {med}")


# ------------ code ------------

items,_=build_items(IN_DIR, csv_file)
model, test_items, hist = train(items)
eval_stopping(model, test_items)
probs = get_probs(model, test_items)
plot_training(hist)
plot_traces(test_items, probs, tau=0.9)
plot_summary(test_items, probs, tau=0.9)