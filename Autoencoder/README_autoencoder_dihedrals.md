# Dihedral Autoencoder (`Autoencoder_dihedral.py`)

Main question to which we are looking for answear is:
> **Do the backbone dihedral angles carry compressible structure, so that a small latent code can reconstruct them?**

---

## Contents

1. [Quick start](#1-quick-start)
2. [Input data](#2-input-data)
3. [How it works](#3-how-it-works)
4. [Memory, disk and time](#4-memory-disk-and-time)
5. [Configuration reference](#5-configuration-reference)
6. [Output files](#6-output-files)
7. [Interpreting the results](#7-interpreting-the-results)
8. [Resuming and re-running](#8-resuming-and-re-running)
9. [Saved models](#9-saved-models)
10. [Limitations](#10-limitations)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick start

Requirements: Python 3.10+, `numpy`, `pandas`, `matplotlib`, `torch`. The script uses MPS (Apple GPU), CUDA, or CPU, whichever is available.

1. Set parameters for your data in CONFIG section
2. **Smoke test first** (a few minutes): set

   ```python
   MAX_TRAIN_FRAMES = 2_000_000
   MAX_TEST_FRAMES  = 100_000
   LATENT_SWEEP     = [8]
   ```

   This checks the data path, the GPU backend and the plots. Results from this run are not scientific (because it is just testing if script works for your data).
   And this tell you how to exacly config script to train it on your data.
3. Look at the printed `frames/s` and `ETA` lines during training (see [section 5](#5-memory-disk-and-time)) and choose `MAX_TRAIN_FRAMES` so the whole sweep fits your time budget.
4. Run the full sweep. The first run also builds the cache, which is the slowest step, and it can be interrupted and restarted.

```bash
python Autoencoder_dihedral.py
```


---

## 2. Input data

One text file per run, for example `dihedrals/<run>/bb_dihedrals.txt`:

```
#      time      b1      g1      d1      e1      z1      a2  ...
       0.00 -129.55   69.93  149.73 -176.51  -76.10  -53.16  ...
      20.00 -174.84   71.41  153.96 -179.85  -85.91  -62.66  ...
```

- The first line starts with `#` and lists column names, the first being `time`.
- Every other line is a frame: time in ps, then dihedral angles **in degrees**.
- Names are letter + residue number (`a` = α, `b` = β, `g` = γ, `d` = δ, `e` = ε, `z` = ζ).
- **Runs may have different dihedral sets.** The union of all names defines the columns; missing entries are padded and masked (see section 4.3).
- The **run label** in tables and plots is the name of the file's folder (`Path(file).parent.name`). Make sure the folder names are unique, or runs with the same name will be merged in the per-run tables (the script prints a warning).

---

## 3. How it works

```
 .txt files ─► cache (.npy, every STRIDE-th frame, parallel) ─► segments (SEG_LEN frames)
                                                                   │
        train: shuffle segments ─► load block (~BLOCK_FRAMES) ─► shuffle frames ─► batches ─► AE
        test : fixed order      ─► load block ─► reconstruct ─► accumulate statistics ─► discard block
```

### 3.1 Cache

Each run is parsed once into `CACHE_DIR/<path>_s<STRIDE>.npy` (float32 degrees, shape `(frames, dihedrals)`, time column removed).

- The parser reads only every `STRIDE`-th line, so a large stride is also fast to build. For `STRIDE = 1` the whole text must be parsed once.
- Files are written to a temporary name and renamed when complete, so an interruption cannot leave a half-written cache file. A cache file that fails to load is detected and rebuilt.
- Cache files are reused only if the **file path and `STRIDE` are identical**. They are **not** refreshed if you edit a source `.txt`; delete the matching `.npy` yourself.
- Cache files made by earlier versions of the in-memory script have the same format and are reused.

### 3.2 Segments and blocks

Every run is cut into **segments** of `SEG_LEN` consecutive frames. A segment is the unit of disk access (one contiguous read of roughly 1 MB).

- **Training:** the segment list is shuffled, segments are collected until `BLOCK_FRAMES` frames are gathered, the block is loaded into RAM, its frames are shuffled, and the model trains on batches from it. Then the block is dropped. Because a block mixes segments from many different runs, batches are diverse even though each segment is contiguous in time.
- **Prefetching:** a background thread loads the next block while the GPU trains on the current one.
- **Testing:** segments are visited in a fixed order, so the same test frames appear in the same order for the baseline and for every latent size. That is what makes per-frame comparisons between latent sizes possible.

### 3.3 cos/sin encoding, padding and masking

- Each angle θ is replaced by `cos θ` and `sin θ`, so the input has `2D` values (`D` = number of columns in the union). This removes the artificial jump at ±180°. Angles are recovered with `atan2(sin, cos)`.
- Missing dihedrals get zeros and a **mask** of 0. The loss and all error statistics ignore masked entries, so padding can never improve the apparent accuracy.
- **Alignment is by column name.** `g15` in one run is assumed to be the same structural dihedral as `g15` in another. This is valid when runs differ only by missing or extra residues at the ends. If the same residue number means something different between runs, interpret results per group of identical headers instead (this script does not have a group mode; use the in-memory script's `GROUP_MODE = "group"` for that).

### 3.4 The model

```
Encoder:  2D ─► 256 ─► 128 ─► latent        (GELU between layers)
Decoder:  latent ─► 128 ─► 256 ─► 2D        (GELU between layers, Tanh output)
```

Masked mean squared error on the cos/sin values, AdamW, cosine learning-rate decay over the whole training budget (stepped per batch).

### 3.5 Training budget

Instead of epochs, each model sees a fixed number of frames:

```
budget = min(TRAIN_PASSES × n_train_frames, MAX_TRAIN_FRAMES)
steps  = budget // BATCH_SIZE
```

`TRAIN_PASSES` can be fractional (for example 0.2 sees a random 20% of the segments' worth of frames). A single pass over hundreds of millions of frames is already a very large amount of training: neighbouring frames are almost identical, so repeating the data many times adds little.

### 3.6 Splits

| Mode | Behaviour |
|---|---|
| `"auto"` | `"run"` if there are at least 5 runs, otherwise `"time"` |
| `"run"` | `TEST_FRAC` of the **runs** go to the test set; the rest train. Preferred: it tests generalisation to unseen simulations. |
| `"time"` | The last `TEST_FRAC` of every run is test, after a gap of `TIME_GAP_FRAMES` (original frames) so train and test do not touch. |
| `"all"` | Train and test on everything. Only a compressibility check; says nothing about generalisation. |

Never split by random frame: it can cause a leak in model training

### 3.7 Baseline

The baseline predicts, for every dihedral, the mean cos and mean sin of the training data. It is estimated from a random `BASELINE_FRAC` of the training segments. An autoencoder only shows structure if it clearly beats this.

### 3.8 Evaluation

Test frames are processed in batches of `EVAL_BATCH`. For each frame the script computes the wrap-safe absolute angular error per dihedral:

```
err = | ((pred − true + 180) mod 360) − 180 |      (179° vs −179° → 2°)
```

and accumulates, without storing the error matrix:

- a histogram of all individual errors (0.25° bins),
- a histogram of per-frame mean errors (0.25° bins),
- per-dihedral error sums, split by per-frame error bin (this gives "which dihedrals are bad in the worst frames"),
- per-run error sums,
- the 20 worst frames with their per-dihedral errors,
- per-frame arrays (error, run, time) for the evaluated frames, used for CSVs and the time plots.

**Which test frames are evaluated.** If the test set has more than `MAX_TEST_FRAMES` frames, every `test_every`-th frame is evaluated (`test_every = ceil(n_test / MAX_TEST_FRAMES)`), evenly across all test runs. **All statistics and plots in the output describe these evaluated frames.** Set `MAX_TEST_FRAMES` higher to evaluate more; memory for the per-frame arrays is about 16 bytes per evaluated frame per latent size.

The mean error is exact. Median, p95 and p99 come from the 0.25° histograms and are accurate to about a quarter of a degree.

---

## 4. Memory, disk and time

### RAM

At startup the script prints an estimate. The main consumer is the block:

```
block ≈ BLOCK_FRAMES × D × 4 bytes        (D = union column count)
```

For `D ≈ 115` and `BLOCK_FRAMES = 1,000,000` a block is about 0.46 GB. Several exist at once (the one being used, its copy on the device, and up to two prefetched), so plan for roughly 2–4 GB plus the per-frame evaluation arrays. This does not grow with the amount of data. Lower `BLOCK_FRAMES` to reduce it; very small blocks mix fewer runs per batch.

On Apple Silicon the GPU shares system memory, so device copies count against the same RAM.

### Disk

The cache holds float32 values, roughly half the size of the text files (4 bytes per value instead of about 8–9 characters; this is an estimate from the file format). You need space for both the text files and the cache. Caches for different `STRIDE` values coexist and are never deleted automatically.

The OS may use free RAM as a file cache for the memory-mapped files. That memory is released automatically when needed and is not a leak.

### Time

The costs are:

1. **First cache build:** one-time. It reads all text from disk and parses every `STRIDE`-th line, using `N_WORKERS` processes.
2. **Training:** `budget / frames_per_second` per latent size. During training the script prints lines like

   ```
   step 1,200/48,000  loss 0.01834  210,000 frames/s  ETA 31.4 min
   ```

   The figures are measured on your machine. Total training time is about `(ETA per model) × number of latent sizes`. Choose `MAX_TRAIN_FRAMES` from that.
3. **Evaluation:** runs on the CPU in NumPy and is roughly proportional to the number of evaluated frames. It is small next to training when `MAX_TEST_FRAMES` is around a million.

If frames/s is lower than you expect, the usual causes are a disk that cannot keep up with block loading (try an SSD, a smaller `STRIDE` range, or a larger `SEG_LEN`) or a small `BATCH_SIZE`.

---

## 5. Configuration reference

| Setting | Example                           | Meaning |
|---|-----------------------------------|---|
| `DATA_GLOB` | `../dihedrals/*/bb_dihedrals.txt` | Pattern for the run files. |
| `STRIDE` | `1`                               | Keep every N-th original frame. `1` = whole data. Part of the cache file name. |
| `dt` | `20.0`                            | Time between original frames (ps). Used only for the time axis: `t = frame_index × STRIDE × dt`. |
| `CACHE_DIR` | `cache_dihedrals`                 | Where parsed `.npy` files go. |
| `N_WORKERS` | CPU count − 2                     | Parallel parsing processes (first pass only). |
| `SPLIT_MODE` | `"auto"`                          | `"auto"`, `"run"`, `"time"` or `"all"`. |
| `TEST_FRAC` | `0.2`                             | Fraction of runs (run mode) or of each run's end (time mode) used as test. |
| `TIME_GAP_FRAMES` | `100`                             | Time mode: gap in original frames before the test part. |
| `LATENT_SWEEP` | `[2, 4, 8, 16, 32, 64]`           | Latent sizes to train and compare. |
| `LATENT_DETAIL` | `8`                               | Latent size used for the detailed plots (the nearest swept size is used if it is not in the sweep). |
| `HIDDEN` | `(256, 128)`                      | Hidden layer widths. |
| `BATCH_SIZE` | `2048`                            | Training batch size. |
| `LR` | `2e-3`                            | AdamW learning rate. |
| `TRAIN_PASSES` | `1.0`                             | Passes over the training frames per model (may be fractional). |
| `MAX_TRAIN_FRAMES` | `100,000,000`                     | Hard cap on frames seen per model. |
| `SEG_LEN` | `2048`                            | Contiguous frames per segment. |
| `BLOCK_FRAMES` | `1,000,000`                       | Frames held in RAM at once. |
| `EVAL_BATCH` | `16384`                           | Evaluation batch size. |
| `MAX_TEST_FRAMES` | `1,000,000`                       | Upper bound on evaluated test frames (frames are thinned evenly beyond this). |
| `BASELINE_FRAC` | `0.1`                             | Share of training segments used for the baseline mean. |
| `HUMP` | `16.0`                            | Per-frame error (deg) used to split "good" and "bad" frames in the analysis. Adjust it to the valley between the two humps in your own histogram. |
| `TOP_PCT` | `99`                              | "Worst frames" = above this percentile of per-frame error. |
| `N_WORST_RUNS` | `4`                               | Runs shown in the error-over-time plot. |
| `RESUME` | `True`                            | Reuse finished latent sizes whose configuration matches. |
| `SEED` | `0`                               | Seed for initialisation, the run split and shuffling. |
| `OUT_DIR` | `../dihedral_ae_out_stream`       | Output folder. |

---

## 6. Output files

Everything goes to `OUT_DIR/all_runs_streamed/`.

| File | Contents |
|---|---|
| `error_vs_latent.png` | Mean and median test error against latent size, with the baseline as a dotted line. |
| `error_histograms.png` | Three panels at `LATENT_DETAIL`: density of all errors (AE vs baseline, three latent sizes), per-frame mean error, and the log-scale tail. |
| `error_per_dihedral.png` | Mean error of each dihedral, sorted from worst to best. |
| `frame_errors_all_latents.csv` | One row per evaluated test frame: run, time (ns), number of dihedrals, and the per-frame mean error for every latent size (`err_L2`, `err_L4`, …). |
| `per_run_error_all_latents.csv` | Mean error of each test run at every latent size, with its dihedral count and number of evaluated frames. |
| `worst_runs_error_vs_time_all_latents.png` | Error over time for the worst runs, one line per latent size. |
| `per_frame_hist_all_latents.png` | Per-frame error histograms for all latent sizes on one axis. |
| `models/ae_latent*.pt` | Saved weights and configuration (see section 10). |
| `eval_baseline.pkl`, `eval_L*.pkl` | Accumulated statistics per model, used for resuming. |

The console output also prints a summary table (mean, median, p95, p99 for the baseline and each latent size) and, for each latent size: the 20 worst evaluated frames with their three worst dihedrals, the share of frames above `HUMP` and the runs dominating that group, the dihedrals that are extra bad in the worst 1% of frames, per-run errors, errors grouped by number of dihedrals, and the overlap (Jaccard index) of the worst frames between latent sizes.

---

## 7. Interpreting the results

### Error vs latent size

- **Gap to the baseline** is the main answer to "is the data worth anything". A large gap means the dihedrals have structure that a small code can capture.
- A curve that **falls and then flattens** shows the effective dimensionality: beyond the flattening point extra latent variables add nothing.
- A curve that is **flat from the first point** means either that very few degrees of freedom already explain the data, or that the remaining error is not limited by the bottleneck (thermal noise, rare flips, too little training). Include small sizes (2, 4) so the elbow is visible.
- A curve that **keeps falling** at the largest size suggests a high-dimensional signal; extend the sweep.
- A mean well above the median means a minority of values is reconstructed very badly (see the tail panel).

### Error histograms

- **All errors, AE vs baseline:** AE curves piled near 0° and far left of the grey baseline indicate real structure. Curves for different latent sizes on top of each other mean extra dimensions did not help.
- **Per-frame error:** one peak means errors are similar across frames. **Two humps** mean a subset of frames or runs is reconstructed several times worse than the rest. Use the per-run table, the number-of-dihedrals table and the time plots to see what that subset is. Do not assume the cause.
- **Tail (log scale):** a plateau at large errors (roughly 50°–125°) followed by a thin tail to 180° is typical of dihedrals that jump between rotamers, which the AE places on the wrong side of the circle. These are rare but raise the mean.

### Per-dihedral errors

High errors are expected for flexible backbone angles (α, γ, ζ, sometimes β and ε), loops and chain ends; low errors for δ (sugar pucker, two main states). If the worst dihedrals cluster at residue numbers that exist in only some runs, suspect fewer training examples for those columns or name-based misalignment.

### Per-run and per-frame analysis

- The **number-of-dihedrals table** shows whether runs with more dihedrals are harder.
- A run whose error **does not fall** as the latent size grows is limited by something other than the bottleneck.
- The **Jaccard overlap** of the worst frames between latent sizes: values near 1 mean the same frames are worst at every size (a structural cause: a rare conformation or flip); values near 0 mean the worst frames change with size (the bottleneck matters).
- **Error over time** shows spikes at conformational events versus a constant offset for poorly represented runs. `frame_errors_all_latents.csv` can be joined to your lifetimes table to test whether the error rises as unfolding approaches.

### Decision guide

| Observation | Likely meaning | What to try |
|---|---|---|
| AE error ≈ baseline | Little structure, or undertrained | Check the training loss, raise `MAX_TRAIN_FRAMES`, lower `STRIDE` |
| AE error ≪ baseline, flat in latent size | Low-dimensional data, or error from noise and rare flips | Add sizes 2 and 4; compare with a PCA baseline and the frame-to-frame noise floor |
| Error keeps falling with latent size | High intrinsic dimension | Extend the sweep |
| Second hump in the per-frame histogram | A subset of runs/frames is poorly reconstructed | Inspect per-run table and time plots |
| Same worst frames at every size | A structural cause | Examine those frames' dihedrals directly |
| Errors concentrated at high residue numbers | Fewer examples or name misalignment | Compare runs by length; check residue counts |

---

## 8. Resuming and re-running

With `RESUME = True`, each finished latent size is saved as `eval_L<size>.pkl` plus its `models/ae_latent<size>.pt`. When the script starts, a size is reused only if the saved **configuration signature** matches. The signature includes `STRIDE`, `TRAIN_PASSES`, `MAX_TRAIN_FRAMES`, `SEED`, `HIDDEN`, `BATCH_SIZE`, `LR`, the number of runs, the column count `D`, the split mode, `TEST_FRAC`, the test thinning factor, and `SEG_LEN`.

Not detected: edited source files, a changed `DATA_GLOB` that happens to give the same number of runs, and `BLOCK_FRAMES`. If you change the data, delete the matching cache files and the `eval_*.pkl` files.

To force retraining, set `RESUME = False` or delete the `eval_*.pkl` files.

---

## 9. Saved models

Each `models/ae_latent<size>.pt` contains `state_dict`, `latent`, `n_in`, `hidden`, `all_names` (the exact column order the model expects) and `stride`. Always use `all_names` when feeding new data: with padded columns the weights are meaningless without it.

Quick look in a notebook (no model class needed):

```python
import torch, pandas as pd
ck = torch.load("dihedral_ae_out_stream/all_runs_streamed/models/ae_latent8.pt", map_location="cpu")
rows = [dict(tensor=k, shape=tuple(t.shape), params=t.numel(), mean=t.mean().item(),
             std=t.std().item(), finite=bool(torch.isfinite(t).all()))
        for k, t in ck["state_dict"].items()]
print(pd.DataFrame(rows).to_string(index=False))
```

All tensors should be finite (`finite=True`); a non-finite tensor means training diverged.

---

## 10. Limitations

- **Statistics describe the evaluated test frames**, which are an even thinning of the test set when it exceeds `MAX_TEST_FRAMES`. Raise `MAX_TEST_FRAMES` to evaluate more.
- **The training loop was not run during development of this script.** The data loading, caching, block construction, evaluation, baseline and reporting were tested on synthetic data with a stand-in for the model. Run the smoke test before a long job.
- **Training is budgeted, not exhaustive.** Each model sees at most `MAX_TRAIN_FRAMES` frames. Results at very small budgets (the smoke test) are not representative.
- **The mean baseline is weak.** It ignores all conformational states. Useful stronger comparisons: PCA on the cos/sin data with the same number of components, and the angular difference between consecutive frames (a thermal-noise floor).
- **Reconstruction is not predictive power.** A small reconstruction error shows the data is compressible; it does not show that the latent code predicts unfolding or lifetimes.
- **Name-based alignment** assumes the same dihedral name means the same structural position across runs.
- **One seed.** Differences of a fraction of a degree between latent sizes are within run-to-run variation.
- **Test frames are correlated in time.** Treat frame-level statistics as descriptive, not as independent samples.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: No files match DATA_GLOB` | Wrong pattern, or a different working directory (the pattern is relative to where you start the script). |
| Windows: the script restarts or errors while parsing | Run it as a script (`python dihedral_ae_stream.py`), not from a notebook or an interactive console. |
| `data columns but … header names` | A file has a malformed header or ragged rows. |
| `non-finite values in …` | A cached run contains NaN/inf; fix the source file and delete its `.npy`. |
| `no data rows after parsing` | A run file has only a header or is empty. |
| `fewer training frames than BATCH_SIZE` | Too little training data for the batch size; reduce `BATCH_SIZE`. |
| Out of memory | Lower `BLOCK_FRAMES` and `MAX_TEST_FRAMES`; close other programs. |
| Very low `frames/s` | Slow disk or small `BATCH_SIZE`. Increase `BATCH_SIZE` (and `LR` slightly), check disk speed. |
| Training loss does not fall | Check the first printed loss lines; try a larger `MAX_TRAIN_FRAMES`, or a smaller `LR`. |
| Reusing old results unexpectedly | `RESUME = True` reuses matching `eval_*.pkl`; delete them or set `RESUME = False`. |
| Duplicate run names warning | Folder names are not unique; per-run tables merge those runs. Rename folders or change the label rule in `main()`. |