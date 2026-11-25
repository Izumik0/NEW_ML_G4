import copy
import glob
import os
import random
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- CONFIG -----------------------------

mode_data = "AE"  # "AE" lub "raw"

AE_weights_path = Path(
    "../results_real/AE_dihedral_261026/dihedral_ae_n_v1/all_runs_streamed/models/ae_latent64.pt"
)

IN_DIR = "../Autoencoder/cache_dihedrals"
OUT_DIR = "../early_warning_test/"
CSV_FILE = "../lifetimes_data_simple.csv"

os.makedirs(OUT_DIR, exist_ok=True)

STRIDE = 25
MAX_N_COLS = 118
FRAMES_PS = 20 * STRIDE  # 500 ps = 0.5 ns na klatkę

H_NS = 10
H_FRAMES = int(H_NS * 1000 / FRAMES_PS)

WINDOW = 20
SAMPLE_STRIDE = 25

BS = 128
DROPOUT = 0.2
HIDDEN_DIM = 128
LAYERS = 2
EPOCHS = 30
LR = 1e-4
PATIENCE = 5
WEIGHT_DECAY = 1e-4
SEED = 42

LEAD_BINS_PS = [
    (0, 10_000),
    (10_000, 25_000),
    (25_000, 50_000),
    (50_000, 100_000),
    (100_000, 250_000),
    (250_000, 500_000),
    (500_000, 1_000_000),
]

NAME_RE = re.compile(
    rf"_(\d{{3}})_([^_]+)_([^_]+)_(\d+)_bb_dihedrals\.txt_s{STRIDE}\.npy$"
)

# ----------------------------- DATA -------------------------------


def load_run(path, target_dihedrals):
    a = np.load(path).astype(np.float32)
    a = np.nan_to_num(a)

    rad = np.deg2rad(a)
    t, n = rad.shape

    if n > target_dihedrals:
        raise ValueError(
            f"Plik zawiera {n} dihedrów, a AE oczekuje "
            f"maksymalnie {target_dihedrals}: {path}"
        )

    x = np.zeros((t, 2 * target_dihedrals), dtype=np.float32)

    x[:, :n] = np.cos(rad)
    x[:, target_dihedrals:target_dihedrals + n] = np.sin(rad)

    return x, n


def parse_names(path):
    m = NAME_RE.search(os.path.basename(path))

    if m is None:
        raise ValueError(f"Nieprawidłowa nazwa pliku: {path}")

    seq, top, lay, copy_id = m.groups()
    return int(seq), top.strip(), lay.strip(), int(copy_id)


def build_items(cache_dir, csv_path, min_unf_ps=2 * H_NS * 1000):
    df = pd.read_csv(csv_path)

    df["seq"] = df["seq"].astype(int)
    df["copy"] = df["copy"].astype(int)
    df["top"] = df["top"].astype(str).str.strip()
    df["lay"] = df["lay"].astype(str).str.strip()

    key = df.set_index(["seq", "top", "lay", "copy"])["unf_time"]

    if not key.index.is_unique:
        raise ValueError("Duplikaty w pliku CSV z metadanymi.")

    items = []
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))

    print(f"Znaleziono {len(files)} plików .npy w {cache_dir}")

    for path in files:
        try:
            run_id = parse_names(path)
        except ValueError:
            continue

        if run_id not in key.index:
            continue

        unf_ps = float(key.loc[run_id])

        if unf_ps < min_unf_ps:
            continue

        items.append((path, unf_ps))

    print(f"Po filtracji zostało {len(items)} runów.")
    return items


def split_runs(items, val_frac=0.15, test_frac=0.20, seed=SEED):
    groups = {}

    for item in items:
        system_id = parse_names(item[0])[:3]
        groups.setdefault(system_id, []).append(item)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    n_test = max(1, int(round(len(keys) * test_frac)))
    n_val = max(1, int(round(len(keys) * val_frac)))

    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test + n_val]
    train_keys = keys[n_test + n_val:]

    def select(selected_keys):
        return [
            item
            for key in selected_keys
            for item in groups[key]
        ]

    return select(train_keys), select(val_keys), select(test_keys)


class EarlyWarningDataset(Dataset):
    """
    Jedna próbka = jedno okno kończące się w czasie t.

    Etykieta:
        1 - zapaść nastąpi w ciągu najbliższych H_NS
        0 - zapaść nie nastąpi w tym horyzoncie

    Dla runów stabilnych (cenzorowanych) pomijamy okna, po których
    nie wiemy, czy zapaść wystąpiłaby w horyzoncie predykcji.
    """

    def __init__(
        self,
        items,
        window_size,
        horizon_frames,
        ae_model,
        device,
        target_dihedrals,
    ):
        self.samples = []
        self.ae_model = ae_model
        self.device = device
        self.target_dihedrals = target_dihedrals

        for path, unf_ps in items:
            x, n_valid = load_run(path, target_dihedrals)
            n_frames = x.shape[0]

            for end in range(window_size, n_frames + 1, SAMPLE_STRIDE):
                start = end - window_size
                window_end_ps = end * FRAMES_PS
                remaining_ps = unf_ps - window_end_ps

                if remaining_ps <= 0:
                    continue

                is_censored = unf_ps >= 1_000_000
                remaining_frames = remaining_ps / FRAMES_PS

                if is_censored and remaining_frames <= horizon_frames:
                    continue

                label = float(remaining_frames <= horizon_frames)

                self.samples.append(
                    {
                        "x": torch.from_numpy(x[start:end]),
                        "label": label,
                        "remaining_ps": remaining_ps,
                        "n_valid": n_valid,
                        "run_path": path,
                        "window_end_ps": window_end_ps,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with torch.no_grad():
            x = sample["x"].unsqueeze(0).to(self.device)

            if self.ae_model is not None:
                x = self.ae_model(x)

            x = x.squeeze(0).cpu()

        return (
            x,
            torch.tensor([sample["label"]], dtype=torch.float32),
            torch.tensor([sample["remaining_ps"]], dtype=torch.float32),
        )


# ----------------------------- MODELS -----------------------------


class AE(nn.Module):
    def __init__(self, n_in, latent, hidden):
        super().__init__()
        h1, h2 = hidden

        self.enc = nn.Sequential(
            nn.Linear(n_in, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, latent),
        )

        self.dec = nn.Sequential(
            nn.Linear(latent, h2),
            nn.GELU(),
            nn.Linear(h2, h1),
            nn.GELU(),
            nn.Linear(h1, n_in),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, x):
        return self.dec(x)

    def forward(self, x):
        return self.enc(x)


def load_ae(path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model = AE(
        checkpoint["n_in"],
        checkpoint["latent"],
        tuple(checkpoint["hidden"]),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    target_dihedrals = int(checkpoint["n_in"] // 2)

    return model, int(checkpoint["latent"]), target_dihedrals


class WindowGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)

        hidden_last_layer = hidden[-1]

        return self.head(hidden_last_layer)


# --------------------------- TRAINING -----------------------------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    probabilities = []
    labels = []
    remaining_times = []

    for x, y, remaining in loader:
        logits = model(x.to(device))

        probabilities.extend(
            torch.sigmoid(logits).cpu().numpy().ravel()
        )
        labels.extend(y.numpy().ravel())
        remaining_times.extend(remaining.numpy().ravel())

    return (
        np.asarray(probabilities),
        np.asarray(labels),
        np.asarray(remaining_times),
    )


def train_model(train_loader, val_loader, input_dim, device):
    model = WindowGRU(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        layers=LAYERS,
        dropout=DROPOUT,
    ).to(device)

    train_labels = np.asarray(
        [sample["label"] for sample in train_loader.dataset.samples]
    )

    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos

    pos_weight = torch.tensor(
        n_neg / max(n_pos, 1.0),
        dtype=torch.float32,
        device=device,
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_left = PATIENCE

    for epoch in range(EPOCHS):
        train_loss = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        val_probs, val_labels, _ = predict(
            model,
            val_loader,
            device,
        )

        if len(np.unique(val_labels)) < 2:
            val_auc = np.nan
        else:
            val_auc = roc_auc_score(val_labels, val_probs)

        print(
            f"Epoka {epoch + 1:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"val AUC: {val_auc:.4f}"
        )

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1

            if patience_left <= 0:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)

    return model, best_val_auc


# ------------------------- EARLY WARNING --------------------------


def auc_by_lead_time(probabilities, labels, remaining_ps):
    rows = []

    for lower, upper in LEAD_BINS_PS:
        mask = (remaining_ps > lower) & (remaining_ps <= upper)

        y = labels[mask]
        p = probabilities[mask]

        if len(np.unique(y)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y, p)

        rows.append(
            {
                "lead_min_ns": lower / 1000,
                "lead_max_ns": upper / 1000,
                "n_samples": int(mask.sum()),
                "n_positive": int(y.sum()),
                "auc": auc,
            }
        )

    return pd.DataFrame(rows)


def collate_with_time(batch):
    xs, ys, remaining = zip(*batch)

    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(remaining),
    )


def save_window_predictions(
    dataset,
    probabilities,
    labels,
    remaining_times,
    output_path,
):
    rows = []

    for sample, probability, label, remaining_ps in zip(
        dataset.samples,
        probabilities,
        labels,
        remaining_times,
    ):
        run_id = parse_names(sample["run_path"])
        run_key = "_".join(map(str, run_id))

        rows.append(
            {
                "run_key": run_key,
                "window_end_ns": sample["window_end_ps"] / 1000,
                "remaining_to_event_ns": remaining_ps / 1000,
                "true_label": int(label),
                "predicted_probability": float(probability),
                "n_valid_dihedrals": sample["n_valid"],
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


# ------------------------------ MAIN ------------------------------

if __name__ == "__main__":
    device = get_device()
    print(f"Urządzenie: {device}")

    items = build_items(IN_DIR, CSV_FILE)

    train_items, val_items, test_items = split_runs(items)

    print(
        f"Runy: train={len(train_items)}, "
        f"validation={len(val_items)}, "
        f"test={len(test_items)}"
    )

    ae_model = None
    latent_dim = 2 * MAX_N_COLS
    target_dihedrals = MAX_N_COLS

    if mode_data == "AE":
        ae_model, latent_dim, target_dihedrals = load_ae(
            AE_weights_path,
            device,
        )

        print(
            f"AE: latent={latent_dim}, "
            f"oczekiwana liczba dihedrów={target_dihedrals}"
        )

    train_dataset = EarlyWarningDataset(
        train_items,
        WINDOW,
        H_FRAMES,
        ae_model,
        device,
        target_dihedrals,
    )

    val_dataset = EarlyWarningDataset(
        val_items,
        WINDOW,
        H_FRAMES,
        ae_model,
        device,
        target_dihedrals,
    )

    test_dataset = EarlyWarningDataset(
        test_items,
        WINDOW,
        H_FRAMES,
        ae_model,
        device,
        target_dihedrals,
    )

    print(
        f"Próbki okien: "
        f"train={len(train_dataset)}, "
        f"validation={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BS,
        shuffle=True,
        collate_fn=collate_with_time,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BS,
        shuffle=False,
        collate_fn=collate_with_time,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BS,
        shuffle=False,
        collate_fn=collate_with_time,
    )

    model, best_val_auc = train_model(
        train_loader,
        val_loader,
        latent_dim,
        device,
    )

    test_probs, test_labels, test_remaining = predict(
        model,
        test_loader,
        device,
    )

    overall_auc = roc_auc_score(test_labels, test_probs)

    lead_table = auc_by_lead_time(
        test_probs,
        test_labels,
        test_remaining,
    )

    print("\n=== WYNIKI EARLY WARNING ===")
    print(f"Ogólny test AUC: {overall_auc:.4f}")
    print(lead_table.to_string(index=False))

    lead_table.to_csv(
        os.path.join(OUT_DIR, "auc_by_lead_time.csv"),
        index=False,
    )

    save_window_predictions(
        test_dataset,
        test_probs,
        test_labels,
        test_remaining,
        os.path.join(OUT_DIR, "test_window_predictions.csv"),
    )

    lead_centers = (
        lead_table["lead_min_ns"]
        + 0.5 * (
            lead_table["lead_max_ns"]
            - lead_table["lead_min_ns"]
        )
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        lead_centers,
        lead_table["auc"],
        marker="o",
    )

    plt.axhline(
        0.5,
        color="gray",
        linestyle="--",
        label="Losowa klasyfikacja",
    )

    plt.xlabel("Czas pozostały do zapaści [ns]")
    plt.ylabel("AUC-ROC")
    plt.title(
        f"Jak wcześnie pojawia się sygnał zapaści?\n"
        f"Okno = {WINDOW * FRAMES_PS / 1000:.1f} ns, "
        f"horyzont = {H_NS} ns"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "auc_vs_lead_time.png"),
        dpi=300,
    )
    plt.show()

    print(
        "\nInterpretacja: wysokie AUC przy dużym czasie pozostałym "
        "do zapaści oznacza, że model wykrywa sygnał nadchodzącej "
        "zapaści odpowiednio wcześnie."
    )