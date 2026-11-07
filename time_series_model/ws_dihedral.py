import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import matplotlib.pyplot as plt
import glob, os
import re
import warnings
from sklearn.metrics import roc_auc_score

# Ignore numpy warnings for cleaner output during training
warnings.filterwarnings("ignore")

# -------- Configuration (Matching your original file) -------------
IN_DIR = "../Autoencoder/cache_dihedrals"
OUT_DIR = "../window_sweep_results/"
csv_file = "../lifetimes_data_simple.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# Data specific config
N_Cols = 118
Frames_ps = 20 * 25  # Time per frame in ps
H_ns = 10  # Prediction horizon in ns
H_frames = int(H_ns * 1000 / Frames_ps)  # Horizon in frames

# Model specific config
bs = 128
drop = 0.2
hidden_dim = 128
layers = 2
Epoches = 10
LR = 1e-4
Patience = 4

# WINDOW SWEEP CONFIGURATION
WINDOW_CANDIDATES = [1, 5, 10, 15, 20, 25, 30, 50] # those are frames (1 frame is Frames_ps)


Name_re = re.compile(r"dihedrals_(\d{3})_([^_]+)_([^_]+)_(\d+)_bb_dihedrals\.txt_s25\.npy$")

# ------------ helpers (Adapted from your original file) --------------

def load_run(path):
    """Loads data and converts angles to cosine/sine representation."""
    a = np.load(path).astype(np.float32)
    T, n = a.shape
    a = np.nan_to_num(a)
    rad = np.deg2rad(a)
    x = np.zeros((T, 2 * N_Cols), np.float32)
    x[:, :n] = np.cos(rad)
    x[:, N_Cols:N_Cols + n] = np.sin(rad)
    return x


def parse_names(path):
    m = Name_re.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"{path} is not a valid file")
    seq, top, lay, copy = m.groups()
    return int(seq), top.strip(), lay.strip(), int(copy)


def build_items(cache_dir, csv_path, min_unf_ps=2 * H_ns * 1000):
    """Loads and filters runs based on CSV metadata."""
    df = pd.read_csv(csv_path)
    df['seq'] = df['seq'].astype(int)
    df['copy'] = df['copy'].astype(int)
    df['top'] = df['top'].astype(str).str.strip()
    df['lay'] = df['lay'].astype(str).str.strip()
    key = df.set_index(["seq", "top", "lay", "copy"])["unf_time"]
    assert key.index.is_unique, "dupicated rows in csv file"

    items = []
    print("Scanning files...")
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.npy"))):
        try:
            k = parse_names(p)
            if k not in key.index:
                continue
            u = float(key.loc[k])
            if u < min_unf_ps:
                continue
            items.append((p, u))
        except ValueError:
            continue

    print(f'Total runs loaded and filtered: {len(items)}')
    return items


def split(items, test_frac=0.2, seed=42):
    """Splits items into training and testing sets based on metadata."""
    groups = {}
    for it in items:
        groups.setdefault(parse_names(it[0])[:3], []).append(it)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = max(1, int(len(keys) * test_frac))
    ts = [it for k in keys[:n_test] for it in groups[k]]
    tr = [it for k in keys[n_test:] for it in groups[k]]
    return tr, ts


# --- Data Handling for Windowing ---
class WindowedDataset(Dataset):
    """
    Loads runs and slices them to the specified window size W.
    """

    def __init__(self, items, window_size):
        self.items = items
        self.window_size = window_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, unf_ps = self.items[i]
        x_full = load_run(path)

        # CRITICAL STEP: Crop the data to the end window (W)
        x_cropped = x_full[-self.window_size:]

        # The label is the binary collapse label for the run
        return torch.from_numpy(x_cropped), torch.tensor([unf_ps < 1_000_000], dtype=torch.float32)


# --- Model Definition (Simplified RNN/GRU) ---
class WindowGRU(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim=64, layers=1, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=layers, dropout=dropout if layers > 1 else 0.0,
                          batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, h_n = self.rnn(x)
        return self.head(h_n.squeeze(0))


# --- Training and Evaluation Logic ---

def run_epoch(model, loader, loss_fun, device, opt=None):
    model.train()
    tot_loss = 0.0
    total_samples = 0.0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = loss_fun(logits, Y).mean()

        if opt:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        tot_loss += loss.item() * Y.mean().item()
        total_samples += Y.mean().item()

    return tot_loss / total_samples


def train_model_for_window(window_size, train_items, test_items, y_true_labels):
    """Trains a single model for a specific window size W."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # --- Data Preparation ---
    train_dataset = WindowedDataset(train_items, window_size)
    test_dataset = WindowedDataset(test_items, window_size)

    dl_tr = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dl_ts = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # --- Loss Function Setup (Handles Imbalance) ---
    neg_samples = sum(1 for label in y_true_labels if label == 0)
    pos_samples = sum(1 for label in y_true_labels if label == 1)
    pos_weight = torch.tensor(neg_samples / max(pos_samples, 1.0), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    # --- Model Initialization ---
    model = WindowGRU(2 * N_Cols, window_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_auc_score = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"  -> Training model for W={window_size}...")

    # --- Training Loop ---
    for ep in range(Epoches):
        run_epoch(model, dl_tr, loss_fn, device, opt=opt)

        # Evaluation phase
        model.eval()
        preds = []
        with torch.no_grad():
            for X, _ in dl_ts:
                logits = model(X)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds.extend(probs)

        # Calculate AUC
        current_auc = roc_auc_score(np.array(y_true_labels), np.array(preds))

        if current_auc > best_auc_score:
            best_auc_score = current_auc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    # Final evaluation with the best model to get probabilities
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in dl_ts:
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)

    return model, preds, roc_auc_score(np.array(y_true_labels), np.array(preds))


def sweep_windows(items, test_items, window_candidates):
    """Runs the core experiment: training models for various window sizes."""

    y_true_labels = [i[1] < 1_000_000 for i in test_items]

    sweep_results = {}
    best_auc = -1.0
    optimal_window = None

    for W in window_candidates:
        # Train the model specific to this window size
        try:
            model, probs, auc = train_model_for_window(W, items, test_items, y_true_labels)
            sweep_results[W] = {'model': model, 'probs': probs, 'auc': auc}
            print(f"--- W={W} Frames | AUC = {auc:.4f} ---")

            if auc > best_auc:
                best_auc = auc
                optimal_window = W
        except Exception as e:
            print(f"!! WARNING: Skipped window W={W} due to error: {e}")

    # --- Reporting and Plotting ---
    print("\n" + "=" * 50)
    print("          WINDOW SWEEP RESULTS")
    print("=" * 50)
    for W, result in sweep_results.items():
        print(f"Window Size W={W} frames: AUC = {result['auc']:.4f}")

    print("\n" + "#" * 50)
    if optimal_window is not None:
        print(f"Optimal Window Size Found: {optimal_window} frames")
        print(f"Maximum Predictive Information (AUC): {best_auc:.4f}")
        print("#" * 50)

        # Plotting the results ONLY if an optimal window was found
        windows = list(sweep_results.keys())
        auc_scores = [sweep_results[W]['auc'] for W in windows]
        plt.figure(figsize=(10, 6))
        plt.plot(windows, auc_scores, marker='o')
        plt.axvline(optimal_window, color='r', linestyle='--', label=f'Optimal W={optimal_window}')
        plt.title("Predictive Information Content vs. Window Size (The Signal)")
        plt.xlabel("Window Size (Frames) / Duration (ps)")
        plt.ylabel("AUC-ROC Score (Information Content)")
        plt.legend()
        plt.grid(True)
        plt.savefig(OUT_DIR + "/auc_vs_window_sweep.png")
        print(f"\nPlot saved to {OUT_DIR}/auc_vs_window_sweep.png")
        plt.show()

        return optimal_window, sweep_results
    else:
        print("⚠️ WARNING: No window found with AUC > 0.5. Prediction signal not found in the tested range.")
        return None, sweep_results  # Return None values gracefully


# --- FUNCTION: SAVING PREDICTIONS FOR ALL WINDOWS ---
def save_predictions_all(sweep_results, test_items):

    all_data = []

    for i, (path, unf_ps) in enumerate(test_items):

        seq, top, lay, cp = parse_names(path)
        run_key = f"{seq}_{top}_{lay}_{cp}"

        # Now, iterate through every window W to get the prediction for this run
        for W, result in sweep_results.items():
            prob_collapse = result['probs'][i]  # Crucial: Indexing matches test_items order
            w_ns = (W*Frames_ps)/1000


            all_data.append({
                "run_key": run_key,
                "unf_ns": unf_ps/1000,
                "stable": unf_ps >= 1_000_000,
                f"prob_ns: {w_ns}": prob_collapse,
            })

    # Convert the list of dictionaries into a single master DataFrame
    df = pd.DataFrame(all_data)

    # Reorder columns for clean readability
    df = df.groupby(["run_key", "unf_ns", "stable"], as_index=False).first()

    output_path = OUT_DIR + "predictions_all_windows.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSUCCESS: Saved comprehensive predictions for all windows to {output_path}")



# ------------ Main Execution Block ------------
if __name__ == "__main__":

    # 1. Load all available runs (Full trace data)
    items = build_items(IN_DIR, csv_file)

    # 2. Split into Train and Test sets
    train_items, test_items = split(items)

    y_true_labels = [i[1] < 1_000_000 for i in test_items]
    print("\n--- LABEL DIAGNOSTICS ---")
    print(f"Total Test Runs: {len(test_items)}")
    print(f"Count of Stable Runs (0): {sum(1 for y in y_true_labels if y == 0)}")
    print(f"Count of Collapsed Runs (1): {sum(1 for y in y_true_labels if y == 1)}")


    # 3. Run the window search (gets the tuple: (optimal_window, sweep_results_dict))
    # We MUST capture the returned tuple into two variables
    optimal_window_result, sweep_results_dict = sweep_windows(train_items, test_items, WINDOW_CANDIDATES)

    # 4. Use the results to generate and save predictions for all windows
    # We pass the dictionary (sweep_results_dict) instead of the tuple
    save_predictions_all(sweep_results_dict, test_items)
