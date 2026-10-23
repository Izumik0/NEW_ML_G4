# Autoencoder compression test for G-quadruplex distance matrices

`Autoencoder_matrix.py` tests whether the 158×158 distance matrices from the simulations can be squeezed through
a small bottleneck and rebuilt without losing meaningful structure.

For each bottleneck size it trains a convolutional autoencoder, tests it on every cached frame, and compares it with
PCA of the same dimension. It reports the error in nm, as variance explained (R²), as a distribution of errors,
over simulation time, per residue pair, and per structural condition. All results are also given for
sequences that were never used for training.

**What it answers:** can the matrix geometry be compressed to *L* numbers with small error, and does that
generalize to unseen sequences?
**What it does not answer:** whether the compressed code keeps the information needed to predict `unf_time`.
That needs a separate test (section 9).

---

## 1. Quick start

Requirements: Python 3, `numpy`, `pandas`, `matplotlib`, `torch` (CUDA, Apple MPS or CPU is chosen automatically).

1. Edit the paths at the top: `CSV`, `NPZ_DIR`, `Cache_dir`, `Out`.
2. Run the script. The first run converts every `.npz` into the cache, which is slow and prints progress.
   Later runs reuse the cache.
3. Training prints one line per epoch. Results appear in the `Out` folder.

Rough cost: each bottleneck size trains for `Epoches × Max_train_frame` frame passes (20 × 100,000 = 2 million by
default), and the sweep has 6 sizes. Time one epoch before a long run. Use `python -u` or `flush=True` on prints,
otherwise console output may appear late and the script can look frozen.

## 2. Inputs

| input | description |
|---|---|
| `CSV` | One row per simulation with columns `seq`, `top`, `lay`, `copy`, `unf_time` (ps). The first column is the index. |
| `NPZ_DIR` | One file per simulation named `distance_arr_<seq>_<top>_<lay>_<copy>.npz`, e.g. `distance_arr_243_-p-l-p_ss_3.npz`. |

Each `.npz` holds one array `(n_frames, 158, 158)` of pairwise distances at 100 ps spacing. If a file holds
several arrays, set `NPZ_key`. Distances are assumed to be in **nm** (the script's reported scale is about 3 nm).
Matrices are computed up to 100 ns (1000 frames), and runs that unfold stop at `unf_time`.

## 3. Settings

| setting | default | meaning |
|---|---|---|
| `Stride` | 10 | Keep every 10th frame, i.e. one frame per ns. 1 keeps every frame (10× the disk and time). |
| `Store_DTYPE` | float32 | Precision of the cache. float16 halves the disk use with about 0.02% relative rounding. |
| `Include_C` | False | Include runs with `unf_time` = 0. |
| `Latents` | 2, 4, 8, 16, 32, 64 | Bottleneck sizes to sweep. |
| `MAP_Latent` | 16 | Bottleneck used for the error map, error-over-time plot and per-condition table. Must be in `Latents`. |
| `Holdout` | 0.2 | Fraction of sequences (at least one) never used for training. |
| `Max_train_frame` | 100000 | Random training frames used to fit each model. |
| `Epoches`, `Batch`, `LR`, `WD` | 20, 32, 1e-3, 1e-4 | Training settings (Adam, cosine learning-rate schedule). |
| `PCA_frames` | 10000 | Training frames used for the scale, the mean structure and PCA. |
| `EVAL_subset` | 1 | Test every n-th frame (1 = all). |
| `BIN_ns` | 10 | Width of the time bins in the error-over-time plot. |
| `SEED` | 42 | Seed for the split and for sampling. |

## 4. How the code works

### 4.1 Run table
The script lists the `.npz` files, parses `seq`, `top`, `lay`, `copy` from each file name and joins them to the CSV on those
four columns. It prints the number of files found, the files with no matching CSV row, and the CSV rows that do
and do not have a file, per group. Runs are put in groups by `unf_time`:

| group | rule | meaning |
|---|---|---|
| A | ≥ 100,000 ps | no unfolding in the matrices (full 1000 frames) |
| B | 0 < `unf_time` < 100,000 ps | run stops at unfolding, fewer frames |
| C | 0 | status unclear, used only if `Include_C = True` |

### 4.2 Split by sequence
`round(Holdout × number of sequences)` sequences (at least 1) are marked `holdout`, the rest `train`. All copies and
all conditions of a held-out sequence stay together, so the held-out error measures generalization to new
sequences, not just to new frames of known runs.

### 4.3 Cache
An `.npz` is a zip archive, so reading one frame means decompressing the whole run. Each run is therefore converted
once to `cache/<name>.npy` containing:

- every `Stride`-th frame (cached position `pos` is original frame `pos × Stride`, time `pos × Stride × DT_PS / 1000` ns);
- only the upper triangle above the diagonal (12,403 values per frame instead of 24,964, since the matrix is symmetric
  with a zero diagonal);
- values of type `Store_DTYPE`.

The file is written under a temporary name and renamed when complete, so an interrupted run leaves no broken file,
and a rerun continues where it stopped. The number of cached frames `m` of each run is read from the file, not from the CSV.
**The cache does not remember its settings: delete the folder after changing `Stride`, `Store_DTYPE` or the source data.**
A relative `Cache_dir` depends on the folder the script is started from.

### 4.4 Frames and sampling
A frame is read from the memory-mapped cache file, divided by `SCALE`, and the full symmetric matrix is rebuilt
(upper triangle filled, transposed copy added, zero diagonal). Items are `(run index, cached position)` pairs.

- Training frames: a random subset of at most `Max_train_frame` frames from the training sequences.
- Test frames: every cached frame of every run (training and held-out), thinned by `EVAL_subset`.

### 4.5 Scale and the mean-structure baseline
From `PCA_frames` random frames of the **training sequences** (held-out sequences are never used) the script computes:

```python
V0 = np.stack([...])                 # (10000, 12403): upper triangle of each sampled frame, in nm
SCALE = float(np.percentile(V0, 99.9))
V0 = torch.from_numpy(V0 / SCALE)    # scaled to about 0-1
MU = V0.mean(0)                      # the mean structure
VAR_TOTAL = float(((V0 - MU) ** 2).mean())
```

- **`SCALE`:** the 99.9th percentile of all sampled distances. Inputs are divided by it. It also converts errors back to nm.
- **`MU` (the mean structure):** for each of the 12,403 residue pairs, the average distance over the sampled frames. It is one
  vector, and rebuilt as a matrix it is the "average matrix" of the sample. All sampled frames count equally, whatever their
  run, condition, group or time, so it is not specific to a condition or a time point.
- **The baseline model:** ignore the input and return `MU` for every frame. It uses no information about the frame at all.
- **`VAR_TOTAL`:** the mean squared error of that baseline on the sample, i.e. the variance of every pair around its average,
  averaged over pairs and frames (scaled units).
- **Printed reference:** `sqrt(VAR_TOTAL) × SCALE` is the RMSE of the baseline in nm, shown as
  "predicting the mean structure gives RMSE ...". An autoencoder error close to it has learned almost nothing. The same
  `VAR_TOTAL` is the denominator of R² (section 6).

Notes on this baseline:

- It is computed on the training sample, not on the test frames. On held-out sequences that differ from the training
  average, the real error of the mean structure is larger than the printed number. The gray baseline in the held-out panel of
  `error_histograms.png` is computed on the held-out frames themselves and is the fair comparison there.
- It is a deliberately weak baseline, because it does not know the condition (`lay`, `top`) or the sequence. A stronger one is
  the mean structure per condition, which shows how much the autoencoder explains beyond what the condition already tells you.
  The script does not compute it.
- PCA is built around the same `MU`: PCA reconstructs `MU` plus a combination of its first *L* components, so PCA with zero
  components is exactly this baseline.

### 4.6 Model (`ConvEncoder`)
Input 1×158×158 → replicate-padded to 160×160 → four stride-2 convolution blocks (16, 32, 64, 128 channels, each
with BatchNorm and ReLU) giving 128×10×10 → linear layer to the bottleneck vector of size *L* → linear layer back to
128×10×10 → three upsample-and-convolve blocks (10 → 20 → 40 → 80) → a final upsample to 160 and a convolution →
cropped to 158×158 → symmetrized as `0.5·(out + outᵀ)`.

### 4.7 Loss
`recon_loss` is the mean squared error over the 12,403 pairs above the diagonal only (the diagonal and the mirrored half
carry no information). It is returned per frame. Training averages it over the batch, and testing keeps one value per frame.

### 4.8 Training
For each bottleneck size a new model is trained with Adam (L2 weight decay `WD`) and a cosine learning-rate schedule over
`Epoches` epochs. There is no validation set and no early stopping: the epoch count is fixed, and generalization is checked
afterwards on the held-out sequences. The model is switched to `eval()` mode for testing (BatchNorm uses its running statistics).

### 4.9 PCA baseline
PCA is fitted once on the centered scaled sample (`torch.pca_lowrank`, 64 components). For every *L* the test frame is
projected on the first *L* components and reconstructed. PCA is the best possible *linear* compression of the same size,
so it shows how much of the autoencoder's performance is nonlinear. PCA errors are computed during the evaluation pass of
the first bottleneck size.

### 4.10 Testing
During the evaluation pass the script stores the per-frame mean squared error, and for `MAP_Latent` it also sums the squared
error of every residue pair over the held-out frames (the error map). Per-frame errors for all models go into the table `meta`
and are saved to `frame_errors.csv.gz`.

### 4.11 Error histograms
After the sweep, the saved models are loaded again and every 5th test frame (`STEP`) is passed through all of them in
one loop. For every residue pair of every frame, the *absolute* distance error is added to a histogram, for the baseline
(the mean structure `MU` from section 4.5, computed here on the test frames) and each bottleneck, on all frames and on held-out sequences only. Errors above `XMAX` (1 nm) are not
drawn, but they are included in the mean shown in the legend.

---

## 5. Outputs (folder `Out`)

| file | content |
|---|---|
| `summary.csv` | Per bottleneck size, method (`enc` or `pca`) and subset (`all`, `train seqs`, `held-out seqs`): `rmse_nm` and `R2`. |
| `error_vs_bottleneck.png` | RMSE and R² against bottleneck size. Solid: autoencoder, dashed: PCA. Black: all, red: held-out. |
| `error_histograms.png` | Distribution of absolute distance errors per bottleneck, with the baseline in gray. |
| `error_vs_time.png` | RMSE over simulation time for `MAP_Latent`, per group. |
| `error_map.npy`, `error_map.png` | Squared error per residue pair, held-out sequences, `MAP_Latent`. |
| `rmse_by_condition.csv` | RMSE per (`lay`, `top`) condition for `MAP_Latent`, sorted. |
| `frame_errors.csv.gz` | One row per test frame: `k` (row of the run table), `pos`, `seq`, `top`, `lay`, `copy`, `group`, `split`, `t_ns`, and the squared errors `enc_<L>` and `pca_<L>` in scaled units. |
| `ae_L<L>.pt` | Weights and `SCALE` for each bottleneck size. |

## 6. Metric definitions

- **Squared errors in `frame_errors.csv.gz`** are in scaled units. To convert a mean to nm: `sqrt(mean) × SCALE`.
- **`rmse_nm`** = `sqrt(mean squared error over the frames) × SCALE`. It is the square root of the average, not the
  average of per-frame RMSEs.
- **`R2`** = `1 − mean squared error / VAR_TOTAL`. R² = 0 means the error equals that of the mean-structure baseline on the training sample (section 4.5), R² = 1 is perfect, and a negative R² is worse than the baseline.
  `VAR_TOTAL` is mainly the variation between runs, so a high R² does not guarantee that the fluctuations *inside* a run
  are reproduced.
- **Histogram legend values** are *mean absolute errors* (MAE). They are smaller than the RMSE values in `summary.csv`
  and cannot be compared with them directly. Compare MAE with the baseline MAE in the same figure.
- Per-frame errors are averaged over all frames of a subset. Runs of different lengths contribute different numbers of frames.

---

## 7. How to read the results

### 7.1 Reference levels
Three numbers give the scale for judging an error:

1. **Mean-structure RMSE** (printed at the start, also the 0.0 R² level): the error of ignoring the frame.
2. **Within-run fluctuation:** how much a pair's distance moves over time inside one run. Compute it from the cache for
   group A runs:
   ```python
   rs = rng.choice(df.index[df.group == "A"], 300, replace=False)
   w = [np.load(paths[k], mmap_mode="r")[:].astype(np.float32).var(axis=0).mean() for k in rs]
   print("within-run RMSE (nm):", np.sqrt(np.mean(w)))        # the cache holds unscaled distances
   ```
3. **Your autoencoder error** (`rmse_nm`).

Reading it: an error far below the within-run fluctuation means individual frames, including their motion, are kept.
An error between the two means the conformation of each run is kept but not the fluctuations. An error near the
mean-structure RMSE means almost nothing was learned.

### 7.2 `summary.csv` and `error_vs_bottleneck.png`
- **Where the curve flattens** is roughly the number of independent degrees of freedom in the data. If it is still falling at
  the largest size, the data needs a larger bottleneck, or the model needs more training. Add sizes like 128 and 256 to find the plateau.
- **Autoencoder against PCA.** If the autoencoder is clearly lower, the data has nonlinear structure and the autoencoder is worth
  using. If the two curves coincide, PCA is enough.
- **Held-out against all.** A gap of up to about 10% is usual. A large gap means the model memorized its training sequences.
  Held-out numbers are noisy when few sequences are held out, so compare them across several seeds before drawing conclusions.
- **A held-out R² near 0 at a very small bottleneck** means the model returns about the mean structure for new sequences: the
  code is too small to describe them. If the same size works for the training sequences (R² clearly above 0), the code only
  learned the training sequences. Repeat with a different `SEED` to check it is not an unlucky split or a failed run.

### 7.3 `error_histograms.png`
It shows how the errors are distributed over individual residue pairs and frames.

- A gray baseline that is much wider than the colored curves means the autoencoder removes most of the error.
- Colored curves from different bottleneck sizes that almost overlap mean larger bottlenecks mostly shrink the long tail, not
  the bulk. A shift of the whole peak to the left means more capacity helps most pairs.
- A long right tail means a few pairs or frames are reconstructed much worse than the rest, which the mean hides.
  Use `error_map.png` to find which pairs they are, and `frame_errors.csv.gz` to find the frames.
- A second bump indicates two populations (for example already-unfolded frames, or particular conditions).
- Compare the *All* and *Held-out* panels: a shift of the held-out curve to the right is the generalization gap.

### 7.4 `error_vs_time.png`
Mean RMSE per 10 ns bin for each group.

- A flat line: all parts of the run are reconstructed equally well.
- A rise: later frames contain structure the model reproduces worse.
- Group B stops at unfolding, so a jump in its last bins can simply be the unfolded structure. Late bins contain few runs, so a spike
  there may come from one or two runs. Bins mix different sets of runs, because runs end at different times.

### 7.5 `error_map.png`
The error per pair of positions, for held-out sequences.

- Bright near the diagonal: local geometry between neighbors is lost.
- Bright far from the diagonal: the large-scale arrangement is lost.
- Uniformly dim: the error is spread evenly.

### 7.6 `rmse_by_condition.csv`
Rows far above the overall RMSE are conditions the bottleneck handles badly: more varied structures, few examples, or many
unfolded frames. Rows far below are conditions with little variation. This table covers all test frames, not only held-out
sequences, and each row is based on a different number of frames, so check the frame count before reading a single row.

### 7.7 Decision guide

| what you see | what it suggests |
|---|---|
| Small error at a small bottleneck, held-out close to all | The geometry is compressible and generalizes. Use that size. |
| Autoencoder and PCA curves identical | Use PCA, it is simpler. |
| Curve still falling at the largest size | Try bigger bottlenecks or more training before concluding. |
| Large held-out gap | Overfitting. More sequences, a smaller model, or more regularization. |
| Long tail in the histogram, a few bright pairs in the map | Information is lost in specific places. Look at which pairs and conditions. |
| Error rising over time | Later frames differ from what the model learns well. Check by group. |

---

## 8. Troubleshooting

| problem | cause and fix |
|---|---|
| Script looks stuck at "cached i/n runs" | The progress prints only every 5 runs and the output may be buffered. Check that `.npy` files are appearing in the cache folder. Use `python -u`. A huge or "online only" synced file can hang one conversion. |
| Training is very slow | Each frame opens a different cache file, which is expensive, especially on Windows. Load the training frames into RAM once, or lower `Max_train_frame` and `Epoches`. Print `device` to confirm a GPU is used. |
| `KeyError: 'enc_2'` or `'ae_2'` | The method name must match in the sweep, the summary and the plot. In this script it is `enc` everywhere. |
| `NaN` in the held-out rows | No held-out frames. Check that at least one sequence is in `holdout`. |
| UserWarning "NumPy array is not writable" | Harmless. It comes from `torch.from_numpy` on a read-only array. |
| Changed `Stride` or `Store_DTYPE` but results look the same | The old cache was reused. Delete the cache folder. |

## 9. Limitations and known issues

- Reconstruction quality is not the same as predictive information. To test whether the compressed code keeps what matters for
  `unf_time`, train a small regressor on the latent vectors and compare it with the same regressor on PCA features or the original inputs.
- The `all` and `train seqs` numbers include frames of training sequences that were not among the `Max_train_frame` frames used to
  fit the model. Only the held-out number is fully independent of training.
- One random split of the sequences (`SEED`) is used. Different seeds can change the held-out numbers.
- Runs have different lengths, so averages over time or conditions mix different sets of runs.
- PCA is fitted on a sample of `PCA_frames` frames, not on the full training set.

Known small issues in the script (they do not change the results, except the first, which only matters for one file layout):

- In `ensure_cache`, the axis check `a.shape[2:] == (N, N)` should be `a.shape[:2] == (N, N)`; as written, files stored as
  `(158, 158, n_frames)` raise an error.
- The print of the scale and the y-axis label of `error_vs_bottleneck.png` say "A", but the values are in nm.
- The function name `eval` shadows a Python built-in. Renaming it to `evaluate` avoids confusion.
- The histogram section reloads the saved models, so it works only after the sweep has finished and written the `ae_L*.pt` files.

## 10. Reproducing a run
Keep `SEED`, `Stride`, `Store_DTYPE`, the data folder and the set of runs the same. The split, the training sample and the
PCA sample are all derived from the seed, so the same settings give the same held-out sequences.