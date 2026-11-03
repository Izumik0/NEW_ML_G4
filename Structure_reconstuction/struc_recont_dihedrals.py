import os
import sys
import re
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")                      # no display needed (nohup / headless Linux)
import matplotlib.pyplot as plt
import seaborn as sns


# ====================================================================
# Config - that can be changed
# ====================================================================

# ----------- Dir config -----------
IN_DIR = Path('../Autoencoder/cache_dihedrals/')
OUT_DIR = Path('../struc_rec_dih_test/')
STRIDE = 25                       # only cache files ending in _s<STRIDE>.npy (or -kopia.npy) are used

# ----------- Data config -----------
N_COLS = 138                      # dihedral columns, shorter runs are zero-padded at the end
frames_train = 500                # frames per training run (upper bound, see max_samples_per_epoch)
max_samples_per_epoch = 200_000   # caps the epoch size when there are many runs
frames_val = 20
frames_test = 150
val_size = 0.1                    # fraction of structures (groups) held out from training for validation
test_size = 0.1                   # fraction of structures (groups) held out for the final test

# ----------- Model config -----------
hidden_dims = 128
batch_size = 256
drop = 0.3
LR = 1e-5
WD = 1e-3
patience = 5
epoches = 10
rnd_state = 69
log_every = 100                   # print progress every N batches

# ---------- Hardware related things -----------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda" and sys.platform.startswith("linux"):
    n_workers = 4
else:
    n_workers = 0

# ====================================================================
# Vocabs and token processing
# ====================================================================
replace_vocab = {'+l+l+l': 'nl_wl_nl',
                 '+l+l+p': 'nl_wl_lp',
                 '+l+p+l': 'nl_sp_wl',
                 '+l+p+p': 'nl_sp_sp',
                 '+ld-l': 'nl_d_wl',
                 '+ld-p': 'nl_d_sp',
                 '+p+l+l': 'lp_nl_wl',
                 '+p+l+p': 'lp_nl_sp',
                 '+p+p+l': 'lp_lp_nl',
                 '+p+p+p': 'lp_lp_lp',
                 '+pd-l': 'lp_d_nl',
                 '+pd-p': 'lp_d_lp',
                 '-l-l-l': 'wl_nl_wl',
                 '-l-l-p': 'wl_nl_sp',
                 '-l-p-l': 'wl_lp_nl',
                 '-l-p-p': 'wl_lp_lp',
                 '-ld+l': 'wl_d_nl',
                 '-ld+p': 'wl_d_lp',
                 '-p-l-l': 'sp_wl_nl',
                 '-p-l-p': 'sp_wl_lp',
                 '-p-p-l': 'sp_sp_wl',
                 '-p-p-p': 'sp_sp_sp',
                 '-pd+l': 'sp_d_wl',
                 '-pd+p': 'sp_d_sp',
                 'd+pd': 'd_sp_d',
                 'd-pd': 'd_lp_d'}
inv_replace = {v: k for k, v in replace_vocab.items()}      # 'wl_lp_lp' -> '-l-p-p'

vocab_top = ['lp', 'sp', 'nl', 'wl', 'd']
vocab_seq = ['1', '2', '3', '4']
vocab = {}
token_id = 0
for t in vocab_top:
    for s in vocab_seq:
        vocab[f'{t}_{s}'] = token_id
        token_id += 1
inverse_vocab = {v: k for k, v in vocab.items()}

alts = sorted(set(replace_vocab) | set(replace_vocab.values()), key=len, reverse=True)
ALT = "|".join(re.escape(a) for a in alts)
SEP = r"[_/\\]"                   # '_' also covers path separators already turned into '_' in the cache name
PAT = re.compile(rf"{SEP}([1-4]{{3}}){SEP}({ALT}){SEP}")


# ====================================================================
# Helpers
# ====================================================================
def labels_from_path(name, vocab):
    """File name -> 3 token ids. Name looks like ..._<seq 3 digits>_<top raw>_...  e.g. _113_-ld+p_"""
    hits = list(PAT.finditer(str(name)))
    if not hits:
        raise ValueError(f"no seq/top found in {name}")
    seq, top_raw = hits[-1].groups()                      # last match = closest to the file name
    top = replace_vocab.get(top_raw, top_raw).split("_")
    return [vocab[f"{t}_{c}"] for t, c in zip(top, seq)]


def decode(ids, inv_vocab):
    """3 token ids -> ('wl_lp_lp', '444')"""
    ids = np.atleast_1d(ids)
    parts = [inv_vocab[int(i)].split("_") for i in ids]
    top = "_".join(p[0] for p in parts)
    seq = "".join(p[1] for p in parts)
    return top, seq


def build_manifest(in_dir, stride):
    """One entry per run. '-kopia' (macOS copy) files collapse onto the original; unparsable files are reported."""
    manifest, skipped = {}, []
    name_re = re.compile(rf"_s{stride}(-kopia)?\.npy$")
    for f in sorted(in_dir.glob("*.npy")):
        if not name_re.search(f.name):
            continue
        is_copy = "-kopia" in f.name
        base = f.name.replace("-kopia", "")
        if is_copy and base in manifest:
            continue                                      # the original is already there
        try:
            labels = labels_from_path(base, vocab)
        except (ValueError, KeyError) as e:
            skipped.append((f.name, str(e)))
            continue
        T, C = np.load(f, mmap_mode="r").shape
        if C > N_COLS:
            skipped.append((f.name, f"{C} columns > N_COLS={N_COLS}"))
            continue
        manifest[base] = {"cache_file": str(f), "labels": labels, "n_frames": int(T), "n_cols": int(C)}
    return manifest, skipped


class DihedralDataset(Dataset):
    """fixed=True : evenly spaced frames of every run, same order every time (val / test, run-major).
       fixed=False: n_samples random (run, frame) pairs per epoch (training)."""

    def __init__(self, runs, manifest, frames, fixed=True, n_samples=None):
        self.runs, self.m, self.n, self.fixed = list(runs), manifest, frames, fixed
        self.T = np.array([manifest[r]["n_frames"] for r in self.runs])
        if fixed:
            self.run_of = np.repeat(np.arange(len(self.runs)), frames)
            self.t_of = np.concatenate([np.linspace(0, T - 1, frames).astype(np.int64) for T in self.T])
            self.length = len(self.run_of)
        else:
            self.length = n_samples if n_samples is not None else len(self.runs) * frames

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.fixed:
            ri, t = int(self.run_of[i]), int(self.t_of[i])
        else:
            ri = int(torch.randint(len(self.runs), (1,)))
            t = int(torch.randint(int(self.T[ri]), (1,)))
        r = self.runs[ri]
        row = np.array(np.load(self.m[r]["cache_file"], mmap_mode="r")[t], dtype=np.float32)   # one row, degrees
        deg = np.zeros(N_COLS, np.float32); deg[:len(row)] = row
        ok = np.zeros(N_COLS, np.float32);  ok[:len(row)] = 1
        rad = np.deg2rad(deg)
        x = np.empty(2 * N_COLS, np.float32)
        x[0::2], x[1::2] = np.sin(rad) * ok, np.cos(rad) * ok                                  # padded angle -> (0, 0)
        return torch.from_numpy(np.nan_to_num(x)), torch.tensor(self.m[r]["labels"], dtype=torch.long)


def make_loaders(manifest, train_runs, val_runs, test_runs, bs=batch_size, workers=n_workers):
    pin = device.type == "cuda"
    n_train = min(len(train_runs) * frames_train, max_samples_per_epoch)
    w_eval = min(workers, 2)
    tr = DataLoader(DihedralDataset(train_runs, manifest, frames_train, fixed=False, n_samples=n_train),
                    batch_size=bs, shuffle=False, num_workers=workers,
                    persistent_workers=workers > 0, pin_memory=pin)
    va = DataLoader(DihedralDataset(val_runs, manifest, frames_val),
                    batch_size=bs, shuffle=False, num_workers=w_eval,
                    persistent_workers=w_eval > 0, pin_memory=pin)
    te = DataLoader(DihedralDataset(test_runs, manifest, frames_test),
                    batch_size=bs, shuffle=False, num_workers=w_eval, pin_memory=pin)
    return tr, va, te


@torch.no_grad()
def eval_loss(model, dl, criterion):
    model.eval()
    tot, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        tot += criterion(model(x), y).item() * len(x)
        n += len(x)
    return tot / n


@torch.no_grad()
def eval_per_run(model, dl, runs, manifest, frames, device):
    """dl must come from a fixed=True dataset built from `runs`, shuffle=False, with `frames` frames per run."""
    model.eval()
    probs = torch.cat([torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in dl])   # (N, 20, 3)
    probs = probs.view(len(runs), frames, 20, 3).mean(1)       # one prediction per run
    conf, pred = probs.max(1)                                  # (n_runs, 3)
    y = torch.tensor([manifest[r]["labels"] for r in runs])
    tok_acc = (pred == y).float().mean().item() * 100
    seq_acc = (pred == y).all(1).float().mean().item() * 100
    two_of_three = ((pred == y).sum(dim=1) == 2).float().mean().item() * 100
    return tok_acc, seq_acc, two_of_three, pred, y, conf


def coverage_report(manifest, train_runs, test_runs):
    seen = [{manifest[r]["labels"][p] for r in train_runs} for p in range(3)]
    ok = [all(t in seen[p] for p, t in enumerate(manifest[r]["labels"])) for r in test_runs]
    print(f"test runs whose 3 tokens all occur in training at that position: {sum(ok)}/{len(ok)}", flush=True)


# ====================================================================
# Model
# ====================================================================
in_dim = 2 * N_COLS


class StructuralPred(nn.Module):
    def __init__(self, in_dim=in_dim, out_dim=3, num_tokens=20, hidden_dim=hidden_dims):
        super().__init__()
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, num_tokens * out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=drop, training=self.training)
        x = F.relu(self.fc2(x))
        return self.fc3(x).view(-1, self.num_tokens, self.out_dim)


# ====================================================================
# Plots and CSV
# ====================================================================
def make_plots(y_true, y_pred, out_dir):
    y_true_flat, y_pred_flat = y_true.reshape(-1), y_pred.reshape(-1)
    vocab_labels = [inverse_vocab[i] for i in range(len(vocab))]
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(len(vocab)))

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=vocab_labels, yticklabels=vocab_labels)
    plt.xlabel('Predicted Token', fontsize=12)
    plt.ylabel('Actual Token', fontsize=12)
    plt.title('Token-Level Confusion Matrix', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    errors = {}
    for i in range(len(vocab)):
        total_actual = cm[i, :].sum()
        errors[inverse_vocab[i]] = ((total_actual - cm[i, i]) / total_actual * 100) if total_actual > 0 else 0.0

    error_df = pd.DataFrame(index=vocab_top, columns=vocab_seq)
    for token_str, error_val in errors.items():
        top, seq = token_str.split('_')
        error_df.loc[top, seq] = error_val
    error_df = error_df.astype(float)

    plt.figure(figsize=(8, 8))
    sns.heatmap(error_df, annot=True, fmt='.1f', cmap='Reds', cbar_kws={'label': 'Error Rate (%)'})
    plt.title('Prediction Error Rate by Token Type (%)', fontsize=16)
    plt.xlabel('Sequence Position (1-4)', fontsize=12)
    plt.ylabel('Topology', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / 'prediction_error.png', dpi=300)
    plt.close()

    print("Raw Error Rates per Token:")
    for token, err in sorted(errors.items(), key=lambda item: item[1], reverse=True):
        print(f"{token:4s}: {err:.2f}% incorrect")


def save_predictions_csv(path, test_runs, y_true, y_pred, conf):
    """One row per run, tokens in threes."""
    rows = []
    for run, t, p, c in zip(test_runs, y_true, y_pred, conf):      # t, p, c: this run's 3 values only
        t_top, t_seq = decode(t, inverse_vocab)
        p_top, p_seq = decode(p, inverse_vocab)
        rows.append({
            "run": run,
            "true_id": " ".join(map(str, t.tolist())),
            "pred_id": " ".join(map(str, p.tolist())),
            "true_top": t_top, "true_top_raw": inv_replace.get(t_top, ""), "true_seq": t_seq,
            "pred_top": p_top, "pred_top_raw": inv_replace.get(p_top, ""), "pred_seq": p_seq,
            "conf_1": round(float(c[0]), 4), "conf_2": round(float(c[1]), 4), "conf_3": round(float(c[2]), 4),
            "tokens_correct": int((t == p).sum()),
            "all_correct": bool((t == p).all()),
            "valid_topology": p_top in inv_replace,         # False if the 3 loop types are not one of the 26 combinations
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ====================================================================
# Main
# ====================================================================
def main():
    print(f'device:{device}, n_workers:{n_workers}', flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- manifest ----------
    manifest, skipped = build_manifest(IN_DIR, STRIDE)
    print(f"{len(manifest)} runs, {len(skipped)} skipped", flush=True)
    for name, why in skipped[:10]:
        print("  skipped:", name[-60:], "|", why)
    assert manifest, f"no usable _s{STRIDE}.npy files in {IN_DIR.resolve()}"

    runs = list(manifest)
    groups = ["-".join(map(str, manifest[r]["labels"])) for r in runs]       # one group per structure
    print(len(set(groups)), "distinct structures", flush=True)

    # ---------- split by structure: test, then validation from the rest ----------
    idx = np.arange(len(runs))
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rnd_state).split(idx, groups=groups))
    trainval_runs = [runs[i] for i in tr]
    test_runs = [runs[i] for i in te]
    g_tv = [groups[i] for i in tr]
    assert len(set(g_tv)) >= 2, "need at least 2 distinct structures in training for a validation split"
    tr2, va2 = next(GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=rnd_state)
                    .split(np.arange(len(trainval_runs)), groups=g_tv))
    train_runs = [trainval_runs[i] for i in tr2]
    val_runs = [trainval_runs[i] for i in va2]
    print(f"train {len(train_runs)} | val {len(val_runs)} | test {len(test_runs)} runs", flush=True)
    coverage_report(manifest, train_runs, test_runs)

    # ---------- loaders, model ----------
    train_loader, val_loader, test_loader = make_loaders(manifest, train_runs, val_runs, test_runs)
    print(f"{len(train_loader)} batches per epoch", flush=True)
    model = StructuralPred().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()

    # ---------- training (early stopping on validation loss) ----------
    best_val, bad_epochs, best_state = float('inf'), 0, None
    t0 = time.time()
    for epoch in range(epoches):
        model.train()
        total = 0.0
        for b, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item()
            if b % log_every == 0:
                print(f"epoch {epoch + 1}/{epoches} batch {b}/{len(train_loader)} "
                      f"loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        train_loss = total / len(train_loader)
        val_loss = eval_loss(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{epoches} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early Stopping", flush=True)
                break

    # ---------- evaluation ----------
    torch.save(best_state, OUT_DIR / "model.pth")
    model.load_state_dict(best_state)

    tok_acc, seq_acc, two_of_three, pred, actual, conf = eval_per_run(
        model, test_loader, test_runs, manifest, frames=frames_test, device=device)
    print(f"Token acc: {tok_acc:.2f}% | Seq acc: {seq_acc:.2f}% | 2 of 3 tokens: {two_of_three:.2f}%", flush=True)

    y_true, y_pred, conf = actual.numpy(), pred.numpy(), conf.numpy()          # (n_runs, 3) each
    make_plots(y_true, y_pred, OUT_DIR)
    save_predictions_csv(OUT_DIR / "predictions.csv", test_runs, y_true, y_pred, conf)
    print("done", flush=True)


if __name__ == "__main__":
    main()