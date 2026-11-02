from collections import Counter
import os
import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from asyncssh import forward
from sklearn import model_selection
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import re
import copy
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit



# Config - that can be changed

# ----------- Dir config -----------
IN_DIR = Path('../cache_dihedrals/')
OUT_DIR = Path('../struc_rec_dih_test/')

# ----------- Data Config -------------
frames_train = 350
frames_test = 75

# ----------- Model Config -----------
hidden_dims = 5
batch_size = 32
drop = 0.3
LR = 1e-5
WD = 1e-3
patience = 5
epoches = 100
rnd_state = 3

# ----------- Dirs things ----------------------
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Hardwear related things -----------
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

# ------------ vocabs and token processing ---------------
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
vocab_top = ['lp', 'sp', 'nl', 'wl', 'd']
vocab_seq = ['1', '2', '3', '4']
vocab = {}
token_id = 0

for t in vocab_top:
    for s in vocab_seq:
        fused = f'{t}_{s}'
        vocab[fused] = token_id
        token_id += 1

# -------- helpers ---------


alts = sorted(set(replace_vocab) | set(replace_vocab.values()), key=len, reverse=True)
ALT = "|".join(re.escape(a) for a in alts)
SEP = r"[_/\\]"                       # '_' also covers path separators already turned into '_' in the cache name

PAT = re.compile(rf"{SEP}([1-4]{{3}}){SEP}({ALT}){SEP}")


def labels_from_path(path, vocab):
    s = str(Path(path).resolve()) if Path(path).exists() else str(path)
    hits = list(PAT.finditer(s))
    if not hits:
        raise ValueError(f"no seq/top found in {path}")
    seq, top_raw = hits[-1].groups()   # last match = closest to the file name
    top = replace_vocab.get(top_raw, top_raw).split("_")
    return [vocab[f"{t}_{c}"] for t, c in zip(top, seq)]

N_COLS = 138
class DihedralDataset(Dataset):
    def __init__(self, runs, manifest, frames, fixed=False):
        self.runs, self.m, self.n, self.fixed = runs, manifest, frames, fixed
        self.T = {r: manifest[r]["n_frames"] for r in runs}
        if fixed:
            self.idx = [(r, int(t)) for r in runs
                        for t in np.linspace(0, self.T[r] - 1, frames)]

    def __len__(self):
        return len(self.idx) if self.fixed else len(self.runs) * self.n

    def __getitem__(self, i):
        if self.fixed:
            r, t = self.idx[i]
        else:
            r = self.runs[i // self.n]
            t = int(torch.randint(self.T[r], (1,)))
        row = np.array(np.load(self.m[r]["cache_file"], mmap_mode="r")[t], dtype=np.float32)
        deg = np.zeros(N_COLS, np.float32); deg[:len(row)] = row
        ok = np.zeros(N_COLS, np.float32);  ok[:len(row)] = 1
        rad = np.deg2rad(deg)
        x = np.empty(2 * N_COLS, np.float32)
        x[0::2], x[1::2] = np.sin(rad) * ok, np.cos(rad) * ok          # padded angle -> (0, 0)
        return torch.from_numpy(np.nan_to_num(x)), torch.tensor(self.m[r]["labels"], dtype=torch.long)

def make_loaders(manifest, train_runs, test_runs, bs=batch_size, workers= n_workers):
 tr = DataLoader(DihedralDataset(train_runs, manifest, frames=frames_train),
                 batch_size=bs, shuffle=True, num_workers=workers,
                 persistent_workers=workers > 0)
 te = DataLoader(DihedralDataset(test_runs, manifest, frames=frames_test))
 return tr, te

def decode(ids, inv_vocab):
    ids = np.atleast_1d(ids)
    parts = [inv_vocab[int(i)].split("_") for i in ids]
    top = "_".join(p[0] for p in parts)
    seq = "".join(p[1] for p in parts)
    return top, seq


@torch.no_grad()
def eval_per_run(model, dl, runs, manifest, frames, device):
    model.eval()
    probs = torch.cat([torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in dl])  # (N, 20, 3)
    probs = probs.view(len(runs), frames, 20, 3).mean(1)       # one prediction per run
    pred = probs.argmax(1)                                    # (n_runs, 3)
    y = torch.tensor([manifest[r]["labels"] for r in runs])
    tok_acc = (pred == y).float().mean().item() * 100
    seq_acc = (pred == y).all(1).float().mean().item() * 100
    count_2_3 = ((pred == y).sum(dim=1) == 2).sum().item()
    count_2_3 = count_2_3 / len(y) * 100
    return tok_acc, seq_acc, count_2_3, pred, y

# ---------- model arch ---------

in_dim = 2 * N_COLS

class StructuralPred(nn.Module):
 def __init__(self, in_dim=in_dim, out_dim=3, num_tokens=20, hidden_dim=hidden_dims, *args: any, **kwargs:any):
  super().__init__(*args, **kwargs)
  self.num_tokens = num_tokens
  self.out_dim = out_dim
  self.fc1 = nn.Linear(in_dim, hidden_dim)
  self.fc2 = nn.Linear(hidden_dim, 32)
  self.fc3 = nn.Linear(32, num_tokens * out_dim)
 def forward(self, x):
  x = F.relu(self.fc1(x))
  x = F.dropout(x, p=drop, training=self.training)
  x = F.relu(self.fc2(x))
  output = self.fc3(x)
  output = output.view(-1, self.num_tokens, self.out_dim)
  return output

# ------- some proper code --------

if __name__=="__main__":
 manifest={}

 for i in sorted(IN_DIR.glob(f"*.npy")):
  y = labels_from_path(i, vocab)
  T,C = np.load(i, mmap_mode="r").shape
  manifest[i.name] = {"cache_file": str(i), "labels": y, "n_frames": T, "n_cols": C}


 runs = list(manifest)
 groups = ["-".join(map(str, manifest[r]["labels"])) for r in runs]
 assert len(groups) == len(runs)
 idx = np.arange(len(runs))
 tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=rnd_state).split(idx, groups=groups))
 assert max(tr) < len(runs) and max(te) < len(runs)
 train_runs = [runs[i] for i in tr]
 test_runs = [runs[i] for i in te]

 train_loader, test_loader = make_loaders(manifest, train_runs, test_runs, bs=batch_size)
 model = StructuralPred()
 optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
 criterion = nn.CrossEntropyLoss()
 model.to(device)

# ----------- Train Loop ------------

best_loss = float('inf')
counter_dd=0
best_so_far=None

for epoch in range(epoches):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_x.to(device))
        loss = criterion(pred, batch_y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epoches} | Training Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        counter_dd = 0
        best_so_far = copy.deepcopy(model.state_dict())
    else:
        counter_dd += 1

    if counter_dd >= patience:
        print("Early Stopping")
        break

# ------- Eval phase -----------
torch.save(best_so_far, OUT_DIR/"model.pth")
model.load_state_dict(best_so_far)

tok_acc, seq_acc, count_2_per, pred, actual = eval_per_run(model, test_loader, test_runs, manifest, frames=frames_test, device=device)
print(f"Token acc: {tok_acc:.4f}, Seq acc: {seq_acc:.4f}")
print(f"2 out of 3 peercent {(count_2_per):.2f} %")
print(f"Predicted tokens: {pred} \nActual tokens: {actual}")

# ----------- Ploting -------------------

y_true_flat = actual.view(-1).numpy()
y_pred_flat = pred.view(-1).numpy()

# 2. Create the reverse vocabulary mapping for our labels
inverse_vocab = {v: k for k, v in vocab.items()}
vocab_labels = [inverse_vocab[i] for i in range(len(vocab))]

# 3. Generate the Confusion Matrix
cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(len(vocab)))

# --- PLOT 1: Confusion Matrix ---
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=vocab_labels, yticklabels=vocab_labels)
plt.xlabel('Predicted Token', fontsize=12)
plt.ylabel('Actual Token', fontsize=12)
plt.title('Token-Level Confusion Matrix', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR/"confiusion_matrix.png", dpi=300)

# 4. Calculate Error Rates per Token
errors = {}
for i in range(len(vocab)):
    total_actual = cm[i, :].sum()
    correct = cm[i, i]

    if total_actual > 0:
        # Percentage of wrong predictions for this specific token
        error_rate = ((total_actual - correct) / total_actual) * 100
    else:
        error_rate = 0.0 # Handle division by zero if a token never appears

    errors[inverse_vocab[i]] = error_rate

# 5. Reshape the errors into a 5x4 DataFrame matching your vocab structure
error_df = pd.DataFrame(index=vocab_top, columns=['1', '2', '3', '4'])

for token_str, error_val in errors.items():
    # Split the token string
    top, seq = token_str.split('_')
    error_df.loc[top, seq] = error_val

# Convert to float for Seaborn compatibility
error_df = error_df.astype(float)

# --- PLOT 2: Error Rate Heatmap ---
plt.figure(figsize=(8, 8))
sns.heatmap(error_df, annot=True, fmt='.1f', cmap='Reds',
            cbar_kws={'label': 'Error Rate (%)'})
plt.title('Prediction Error Rate by Token Type (%)', fontsize=16)
plt.xlabel('Sequence Position (1-4)', fontsize=12)
plt.ylabel('Topology', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'prediction_error.png', dpi=300)

# Optional: Print raw error rates for quick reading
print("Raw Error Rates per Token:")
for token, err in sorted(errors.items(), key=lambda item: item[1], reverse=True):
    print(f"{token:4s}: {err:.2f}% incorrect")

# ------------- Final .csv file making --------------------

rows=[]
y_true_flat = np.asarray(y_true_flat).reshape(-1, 3)      # (n_runs, 3)
y_pred_flat = np.asarray(y_pred_flat).reshape(-1, 3)

for p, ac in zip(y_true_flat, y_pred_flat):
    true_top, true_seq, pred_top, pred_seq = [], [], [], []
    for pos in range(3):
        t_top, t_seq = decode(ac[pos], inverse_vocab)
        p_top, p_seq = decode(p[pos], inverse_vocab)

        true_top.append(t_top)
        true_seq.append(t_seq)
        pred_top.append(p_top)
        pred_seq.append(p_seq)

    rows.append({
        "true_id": y_true_flat.tolist(), "pred_id": y_pred_flat.tolist(),
        "true_top": true_top, "true_seq": true_seq,
        "pred_top": pred_top, "pred_seq": pred_seq,
    })

final = pd.DataFrame(rows)
final.to_csv(OUT_DIR / 'predictions.csv', index=False)
