# Dihedral Autoencoder: Is the Backbone Dihedral Data Worth Anything?

An autoencoder (AE) test for G-quadruplex (G4) molecular-dynamics trajectories. It answers one question:

> **Do the backbone dihedral angles contain structured, compressible information, so that a small number of latent variables can reconstruct them?**

If an AE with a latent code of, say, 8 numbers can rebuild ~100 dihedral angles with a small error, the data has low effective dimensionality and real structure. If the AE barely beats a trivial "predict the average" baseline, the dihedrals are mostly noise (or the model is too weak or undertrained).

---

## Contents

1. [Quick start](#1-quick-start)
2. [Input data format](#2-input-data-format)
3. [How the pipeline works](#3-how-the-pipeline-works)
4. [The model](#4-the-model)
5. [Train/test splits](#5-traintest-splits)
6. [Metrics](#6-metrics)
7. [Output files](#7-output-files)
8. [How to interpret the results](#8-how-to-interpret-the-results)
9. [Configuration reference](#9-configuration-reference)
10. [Saving and inspecting model weights](#10-saving-and-inspecting-model-weights)
11. [Caveats and limitations](#11-caveats-and-limitations)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Quick start

Requirements: Python 3.10+, `numpy`, `pandas`, `matplotlib`, `torch` (the script picks MPS on Apple Silicon, CUDA if available, otherwise CPU).

1. Edit the **CONFIG** block at the top of the script (`DATA_GLOB`, `STRIDE`, `dt`, `EPOCHS`, `LATENT_SWEEP`).
2. **Smoke test first**: set `EPOCHS = 5`, `LATENT_SWEEP = [8]`, `STRIDE = 50`. This checks that data loading, masking and plotting work. The errors at 5 epochs are not scientific results.
3. Run the full experiment: `EPOCHS = 40–60`, `LATENT_SWEEP = [2, 4, 8, 16, 32, 64]`.
4. Open the figures and CSVs in `OUT_DIR/all_runs_padded/`.

```bash
python Autoencoder_dihedral.py
```

---

## 2. Input data format

One text file per simulation run (e.g. `dihedrals/<run>/bb_dihedrals.txt`):

```
#      time      b1      g1      d1      e1      z1      a2  ...
       0.00 -129.55   69.93  149.73 -176.51  -76.10  -53.16  ...
      20.00 -174.84   71.41  153.96 -179.85  -85.91  -62.66  ...
```

- The first line starts with `#` and lists the column names. The first name is `time`.
- Every following line is one frame. The first number is time (ps), the rest are dihedral angles **in degrees**, in (−180, 180].
- Frame spacing is **20 ps** (`dt = 20.0` in the config).
- Column names are a letter plus a residue number: `a` = α, `b` = β, `g` = γ, `d` = δ, `e` = ε, `z` = ζ. For example `g15` is γ of residue 15.
- **Runs may have different numbers of dihedrals** (different sequences or lengths). The script handles that (see [padding and masking](#padding-and-masking)).
- No NaNs are allowed. The script raises an error if it finds any.

`time` is dropped on loading. A time axis is rebuilt later as `(frame index) × STRIDE × dt`.

Parsed files are cached as `.npy` in `CACHE_DIR`. **Delete the cache whenever you change `STRIDE` or replace the source files** (the stride is part of the cache file name, but changed source data is not detected).

---

## 3. How the pipeline works

```
 .txt files ──► load + stride ──► align columns by name (union) ──► split ──► cos/sin ──► AE ──► decode
                                                                                              │
 error figures ◄── angular error (degrees, wrap-safe) ◄── arctan2(sin, cos) ◄─────────────────┘
```

### 3.1 Why cos/sin and not raw angles

A dihedral is a circular quantity: −179° and +179° are only 2° apart, but numerically they are 358 apart. Feeding raw degrees to a network creates an artificial discontinuity at ±180°, and an MSE loss would heavily punish a reconstruction that is physically almost perfect.

Each angle θ is therefore replaced by two numbers, **cos θ and sin θ**. Both lie in [−1, 1], the representation is continuous around the circle, and no angle is privileged. For D dihedral columns the network input has **2D** values, arranged as `[cos₁ … cos_D | sin₁ … sin_D]`.

To get an angle back, the script computes `θ = atan2(sin, cos)`. This tolerates network outputs that are not exactly unit-length, because only the direction matters.

### 3.2 Padding and masking

Runs with different dihedral sets are combined by taking the **union of all column names** (sorted by residue, then by α, β, γ, δ, ε, ζ):

- A run that has a column keeps its real data there.
- A run that lacks a column gets zeros there (cos = sin = 0, which is not a valid angle) and a **mask** value of 0.
- The loss is computed **only over mask = 1 entries**, so the model is never rewarded or punished for padding.
- All error statistics skip masked entries (they are NaN internally and ignored with `nanmean`/`nanmedian`).

This means padded zeros can never make the error look better than it is.

> **Important:** alignment is by column *name*. `g15` in one run is assumed to be structurally the same dihedral as `g15` in another run. That is true if the runs differ only by missing or extra end residues. If the runs are different sequences or topologies where the same residue number means something different, use `GROUP_MODE = "group"`, which trains a separate AE for each identical header set.

### 3.3 Stride

`STRIDE = N` keeps every N-th frame. With `dt = 20 ps`, `STRIDE = 5` gives 100 ps spacing and `STRIDE = 50` gives 1 ns. Neighbouring 20 ps frames are nearly identical, so a larger stride makes the dataset smaller and faster without losing much information. It also makes the test frames less correlated with one another.

---

## 4. The model

A fully connected autoencoder with a bottleneck.

```
Encoder:  2D ──► 256 ──► 128 ──► latent
              GELU    GELU
Decoder:  latent ──► 128 ──► 256 ──► 2D
                  GELU    GELU    Tanh
```

- **Input / output size 2D**, where D is the number of columns in the union (including padded ones).
- **Latent size** is the bottleneck. It is the number being swept (`LATENT_SWEEP`). It is the number of values used to describe one frame.
- **GELU** hidden activations: smooth, avoids dead units, a sensible default for smooth inputs like cos/sin. ReLU or SiLU would probably give similar results.
- **Tanh output**: cos and sin lie in [−1, 1], so the decoder is bounded to the same range.
- **Loss:** masked mean squared error on the cos/sin values, `sum(mask · (x̂ − x)²) / sum(mask)`.
- **Optimizer:** AdamW, learning rate `LR`, cosine-annealed to zero over `EPOCHS`.

The AE is intentionally small and plain. The question is about the data, not about finding the best architecture.

---

## 5. Train/test splits

`SPLIT_MODE` controls what the model is tested on. Test error is always measured on data the model did not train on, except in `"all"` mode.

| Mode | What it does | Use when |
|------|--------------|----------|
| `"run"` | Whole runs go to train or test (`TEST_FRAC` of the runs). | **Preferred** with 5+ runs. Tests generalisation to unseen simulations. |
| `"time"` | The last `TEST_FRAC` of every run is test; a gap of `TIME_GAP_FRAMES` is dropped before it. | Few runs (e.g. one file). Tests generalisation to later parts of the same trajectory. |
| `"all"` | Train and test on everything. | A pure compressibility check ("is information conserved through the bottleneck?"). It says nothing about generalisation. |
| `"auto"` | `"run"` if there are at least 5 runs, else `"time"`. | Default. |

**Never split randomly by frame.** Frames 20 ps apart are almost identical, so a random frame split leaks the test set into training and makes the results look much better than they are.

`TIME_GAP_FRAMES` is in original 20 ps frames and is divided by `STRIDE` internally.

---

## 6. Metrics

All errors are reported in **degrees** after converting the reconstruction back with `atan2`.

**Wrap-safe angular error**

```
err = | ((pred − true + 180) mod 360) − 180 |
```

So the error between 179° and −179° is 2°, not 358°. The maximum possible error is 180°.

**Baseline ("predict the mean")**

For every dihedral, the mean cos and mean sin over the training set are computed, and every frame is "reconstructed" as that constant. This is the error you get from knowing nothing about each frame. An autoencoder only demonstrates structure if it clearly beats this.

**Reported statistics:** mean, median, 95th and 99th percentile over all test values. The median is the typical error; the mean is pulled up by rare large errors; the percentiles describe the tail.

---

## 7. Output files

Everything is written to `OUT_DIR/all_runs_padded/` (or `groupK_…` folders in group mode).

| File | Contents |
|------|----------|
| `error_vs_latent.png` | Mean and median test error vs latent size, with the baseline as a dotted line. |
| `error_histograms.png` | error distribution of all values (AE vs baseline)|
| `error_per_dihedral.png` | Mean error of each dihedral at `LATENT_DETAIL`, sorted from worst to best. |
| `frame_errors_all_latents.csv` | One row per test frame: run, time (ns), number of dihedrals, and mean error for every latent size (`err_L2`, `err_L4`, …). |
| `per_run_error_all_latents.csv` | Mean error of every run at every latent size, with the run's dihedral count and number of test frames. |
| `worst_runs_error_vs_time_all_latents.png` | Error over time for the four worst runs, one line per latent size. |
| `per_frame_hist_all_latents.png` | Per-frame error histograms for all latent sizes on one axis. |
| `models/ae_latent*.pt` | Saved weights and configuration (only if checkpoint saving is added, see [section 10](#10-saving-and-inspecting-model-weights)). |

The console output also prints, for every latent size: the 20 worst frames with their three worst dihedrals, the share of frames in the high-error group and the runs that dominate it, the dihedrals that are most over-represented in the worst 1% of frames, per-run errors, errors grouped by number of dihedrals, and an overlap matrix of the worst frames between latent sizes.

---

## 8. How to interpret the results

### 8.1 Error vs latent size (`error_vs_latent.png`)

This is the headline result.

- **Gap to the baseline.** The larger the gap between the AE curve and the dotted baseline line, the more structure the dihedrals have. In one early test, the baseline was about 30.6° and the AE reached about 13° with a median near 8° at latent 8–16. That means 8 numbers captured much of what ~100 angles contain. (Early-test numbers; your values depend on stride, epochs and data.)
- **Shape of the curve.**
  - Falls steeply, then flattens: the flattening point is the *effective dimensionality* of the data. Beyond it, more latent variables add nothing.
  - Flat from the start (e.g. 8 and 16 give the same error): either the data already has ≤8 effective degrees of freedom, or the remaining error comes from something the bottleneck does not limit (thermal noise, rare flips, undertraining).
  - Keeps falling at the largest latent size: the data is high-dimensional, or the AE is acting as a near-identity map. Try larger sizes.
- **Mean above median.** Typical when a small fraction of values are reconstructed very badly. The tail panel shows how.

### 8.2 Error histograms (`error_histograms.png`)

**all values, AE vs baseline.**
AE curves piled up near 0° and well left of the grey baseline = real structure. Curves for different latent sizes lying on top of each other = extra latent dimensions do not help.

### 8.3 Error per dihedral (`error_per_dihedral.png`)

- **High error** is typical for the flexible backbone angles (α, γ, ζ, sometimes β and ε), especially in loops and terminal residues (chain ends fray).
- **Low error** is typical for δ, which follows sugar pucker and has two well-defined states.
- If the worst dihedrals cluster at **high residue numbers** that exist in only some runs, that is a sign of two possible causes: fewer training examples for those columns, or a name-alignment problem (the same name meaning different structural positions in different runs). Compare with `GROUP_MODE = "group"`.

### 8.4 Per-frame and per-run analysis

Use `frame_errors_all_latents.csv` and `per_run_error_all_latents.csv`.

- **Worst-frames table (console):** the `worst_3_dihedrals` column. If the same dihedrals recur, the failure is localised (a flip or a flexible loop). If they vary, the errors are diffuse.
- **Error by number of dihedrals:** if runs with more dihedrals have clearly higher error, the padded approach is a likely culprit.
- **Overlap matrix (Jaccard) of the worst 1% frames:**
  - Near 1: the same frames are the worst at every latent size. They are hard for a *structural* reason (a rare conformation, a flip, an unusual state) and a bigger bottleneck will not fix them.
  - Near 0: the worst frames change with latent size, so the bottleneck matters and errors are more random.
- **Error vs time:** a spike or step at a particular time marks a conformational event; a constantly elevated line means the whole run is poorly represented. You can join the CSV with the lifetimes table (`lifetimes_data_simple.csv`) to test whether reconstruction error rises as unfolding approaches. If it does, the error itself is informative.
- **Per-run table across latent sizes:** a run whose error does *not* fall as the latent size grows is limited by something other than the bottleneck.

### 8.5 Decision guide

| Observation | Likely meaning | What to try |
|-------------|----------------|-------------|
| AE error ≈ baseline | Little structure, or undertrained. | More epochs, lower stride, larger latent. Check the training loss. |
| AE error ≪ baseline, flat in latent size | Low-dimensional data, or error dominated by noise or rare flips. | Add latent sizes 2 and 4; compare to a PCA baseline and to the frame-to-frame noise floor. |
| AE error falls steadily with latent size | High intrinsic dimension. | Extend the sweep to larger sizes; consider a deeper model. |
| Test error ≫ train error | Overfitting or an unrepresentative split. | Use `"run"` split, more runs, regularisation. |
| Second hump in per-frame histogram | A subset of runs or frames is poorly reconstructed. | Inspect per-run CSV and error vs time; try group mode. |
| Same frames worst at all latent sizes | A structural cause (rare conformation, flips). | Look at those frames' dihedrals directly. |
| Errors concentrated at high-numbered residues | Fewer examples, or name misalignment. | Group mode; check residue counts per run. |

---

## 9. Configuration reference

| Setting | Meaning |
|---------|---------|
| `DATA_GLOB` | File or glob pattern for the run files. |
| `STRIDE` | Keep every N-th frame (1 = every 20 ps). |
| `dt` | Time between original frames in ps (20.0). Only used to build the time axis. |
| `CACHE_DIR` | Folder for parsed `.npy` files. Delete after changing source data. |
| `GROUP_MODE` | `"pad"` = one model with name-aligned padded columns; `"group"` = one model per identical column set. |
| `MIN_RUNS_PER_GROUP` | Group mode only: skip groups with fewer runs. |
| `SPLIT_MODE` | `"auto"`, `"run"`, `"time"` or `"all"` (see section 5). |
| `TEST_FRAC` | Fraction of runs (run mode) or of each run's end (time mode) used as test. |
| `TIME_GAP_FRAMES` | Time mode: gap between train and test, in original frames. |
| `LATENT_SWEEP` | Latent sizes to train and compare. |
| `LATENT_DETAIL` | Latent size shown in the detailed histograms and per-dihedral plot. |
| `HIDDEN` | Hidden layer widths, `(256, 128)`. |
| `EPOCHS` | Training epochs (use 5 only for smoke tests). |
| `BATCH_SIZE` | Batch size. Smaller is slower per epoch, especially on MPS. |
| `LR` | AdamW learning rate. |
| `SEED` | Seed for model initialisation and the run split. |
| `OUT_DIR` | Where figures and CSVs are written. |
| `SHOW_PLOTS` | `True` to also open the figures; call `plt.close(fig)` after saving if many figures stay open. |

---

## 10. Saving and inspecting model weights

The script does not save models unless you add checkpointing inside the sweep loop:

```python
ckpt_dir = out / "models"; ckpt_dir.mkdir(exist_ok=True)
torch.save({
    "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
    "latent": latent, "n_in": Xtr.shape[1], "hidden": HIDDEN,
    "all_names": all_names,      # column order the model expects
    "stride": STRIDE, "epochs": EPOCHS,
}, ckpt_dir / f"ae_latent{latent}.pt")
```

Always store `all_names` with the weights: with padded columns, the weights are meaningless without the exact column order.

**Quick look in a notebook (no model class needed):**

```python
import torch, pandas as pd
ck = torch.load("dihedral_ae_out/all_runs_padded/models/ae_latent8.pt", map_location="cpu")
rows = [dict(tensor=k, shape=tuple(t.shape), params=t.numel(), mean=t.mean().item(),
             std=t.std().item(), finite=bool(torch.isfinite(t).all()))
        for k, t in ck["state_dict"].items()]
print(pd.DataFrame(rows).to_string(index=False))
```

**Sanity checks:** all tensors must be finite; weights should have means near 0 and standard deviations of roughly 0.01–0.3; a model rebuilt from the file must reproduce the in-memory reconstructions (maximum difference ~1e-6 or smaller).

**Loading for later use:**

```python
model = AE(ck["n_in"], ck["latent"]).to(device)
model.load_state_dict(ck["state_dict"]); model.eval()
```

New data must use the same column order (`ck["all_names"]`) and the same cos/sin layout.

---

## 11. Caveats and limitations

- **The mean baseline is weak.** It ignores all conformational states, so beating it is necessary but not impressive on its own. Two stronger comparisons are cheap and informative:
  - **PCA on the cos/sin data** with the same number of components. If PCA matches the AE, nonlinearity is not adding much.
  - **A noise floor** from the angular difference between consecutive frames (20 ps apart). If the AE median error is close to it, the remaining error is thermal fluctuation, not information loss.
- **Reconstruction is not the same as predictive power.** A good reconstruction shows the data is compressible. It does not show that the latent code predicts unfolding or lifetimes; that needs a separate model.
- **Name-based alignment** assumes the same name means the same structural position across runs.
- **Undertrained models** (very few epochs) overstate the error and can hide the elbow in the latent curve.
- **One seed.** Differences of a fraction of a degree between latent sizes are within run-to-run variation. Repeat with a few seeds before reading anything into small differences.
- **Test frames are correlated** in time. Treat the frame-level statistics as descriptive, not as independent samples.
- **Cache staleness.** Changing the source files without clearing `CACHE_DIR` silently reuses old data.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `FileNotFoundError: No files match DATA_GLOB` | Wrong pattern or working directory. The path is relative to where you run the script. |
| `data columns but … header names` | A file has a malformed header or ragged rows. |
| `contains NaN` | Clean the file before use. |
| `KeyError: 'n_dihedrals'` (or other missing column) | The frame table must be built with the `n_dihedrals` column included. |
| `NameError: te_run` / `te_t` | The split must use the four-value `stack(...)` and unpack `te_deg, te_pres, te_run, te_t`. |
| Unpacking error in `GROUP_MODE = "group"` | Group code must carry `(array, names, file)` triples, not pairs. |
| `Pandas4Warning … mean(1)` | Use `mean(axis=1)`. It is only a deprecation warning. |
| Many figures kept open / memory warning | Call `plt.close(fig)` after each `savefig`. |
| Very slow epochs | Increase `STRIDE` or `BATCH_SIZE`; use MPS/CUDA if available. |
| Errors much worse than expected | Check `EPOCHS`; 5 epochs is a smoke test only. |
