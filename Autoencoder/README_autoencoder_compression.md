# Autoencoder compression test for G-quadruplex distance matrices

`autoencoder_compression.py` tests whether the 158×158 distance matrices of the simulations can be
compressed through a small bottleneck and reconstructed without losing meaningful structure.

For each bottleneck size it trains a convolutional autoencoder, tests it on all cached frames, and
compares it with PCA of the same dimension. Errors are reported in angstrom and as variance explained (R²),
both on all data and on sequences that were never used for training.

## 1. Inputs

| input | description |
|---|---|
| `CSV` | Table with one row per simulation and columns `seq`, `top`, `lay`, `copy`, `unf_time` (ps). The first column is the index. |
| `NPZ_DIR` | Folder with one file per simulation, named `distance_arr_<seq>_<top>_<lay>_<copy>.npz`, e.g. `distance_arr_243_-p-l-p_ss_3.npz`. |

Each `.npz` must contain one array of shape `(n_frames, 158, 158)` (or `(158, 158, n_frames)`, which is reordered).
If a file holds several arrays, set `NPZ_key` to the array name.

Frame spacing can be ajusted (`DT_PS`). Matrices are computed up to point of unfold stop at `unf_time`.

## 2. Pipeline

1. **Match files to the table.** The script lists the `.npz` files, parses `seq`, `top`, `lay`, `copy` from each name
   and joins them to the CSV. It prints the number of files found, files without a CSV row, and CSV rows with a file
   per group.
2. **Groups.** A: `unf_time` ≥ 100,000 ps (no unfolding in the matrices). B: 0 < `unf_time` < 100,000 ps (run stops at
   unfolding). C: `unf_time` = 0, used only if `Include_C = True`. (Groups are used in early stage of autoencoder development)
3. **Split by sequence.** A fraction `Holdout` of the sequences is held out entirely (all copies, all conditions of
   that sequence). The held-out sequences are used only for testing.
4. **Cache.** Each `.npz` is decompressed once and saved to `cache/<name>.npy` (see section 4).
5. **Scale.** Distances are divided by `SCALE`, the 99.9th percentile of a random sample of training frames.
6. **Training.** For each bottleneck size in `Latents`, a new autoencoder is trained on a random subset of
   `Max_train_frame` frames from the training sequences (Adam, cosine learning-rate schedule, `Epoches` epochs,
   no early stopping).
7. **Testing.** Every cached frame (all runs, including held-out sequences) is passed through the model and its
   reconstruction error is stored (`EVAL_subset` > 1 tests every n-th frame only).
8. **PCA baseline.** PCA is fitted on the same sample as the scale (`PCA_frames` frames) and evaluated with the first
   *L* components for every *L* in `Latents`.
9. **Outputs.** Summary table, plots and per-frame errors (section 5).

## 3. Model and loss

- Input: the symmetric matrix with a zero diagonal, padded from 158 to 160.
- Encoder: four stride-2 convolution blocks (16, 32, 64, 128 channels, BatchNorm, ReLU), giving 128×10×10, then a
  linear layer to the bottleneck of size *L*.
- Decoder: a linear layer back to 128×10×10, three upsample and convolution blocks, a final upsample and convolution,
  cropped to 158×158.
- The output is symmetrized: `0.5 · (out + outᵀ)`.
- Loss: mean squared error over the 12,403 pairs above the diagonal only. The diagonal and the mirrored half carry
  no information and are excluded. The per-sample loss is averaged over the batch for training and kept per frame for testing.

## 4. Cache

`.npz` files cannot be read frame by frame, so each run is converted once:

- only every `Stride`-th frame is kept (`Stride = 10` is one frame per ns);
- only the upper triangle is stored (12,403 values per frame instead of 24,964);
- values are stored as `Store_DTYPE` (float16 by default, about 0.02% relative rounding; use `np.float32` to avoid it).

Files are written to a temporary name and renamed, so an interrupted conversion leaves no half-written file.
Existing cache files are reused. **After changing `Stride`, `Store_DTYPE` or the source data, delete the cache
folder.** The first run is slow, because every `.npz` has to be read and decompressed. A cached frame is about 25 KB
in float16.

During training and testing a frame is read from the memory-mapped `.npy` file and the full symmetric matrix is rebuilt.

## 5. Outputs (folder `Out`)

| file | content |
|---|---|
| `summary.csv` | For each bottleneck size, method (`ae` or `pca`) and subset (`all`, `train seqs`, `held-out seqs`): `rmse_A` and `R2`. |
| `frame_errors.csv.gz` | One row per tested frame: run identifiers, `group`, `split`, time in ns, and the mean squared error (scaled units) for `ae_<L>` and `pca_<L>`. |
| `error_vs_bottleneck.png` | RMSE (Å) and R² against bottleneck size, autoencoder (solid) and PCA (dashed). |
| `error_vs_time.png` | RMSE over simulation time for the bottleneck `MAP_Latent`, per group. |
| `error_map.npy`, `error_map.png` | Mean squared error per residue pair over held-out sequences, for `MAP_Latent`. |
| `rmse_by_condition.csv` | RMSE per (`lay`, `top`) condition for `MAP_Latent`. |
| `ae_L<L>.pt` | Model weights and `SCALE` for each bottleneck size. |

Definitions:

- `rmse_A` = √(mean squared error) × `SCALE`, the typical error of a reconstructed distance in Å.
- `R2` = 1 − mean squared error / `VAR_TOTAL`, where `VAR_TOTAL` is the variance of the training sample around its mean
  structure. R² = 0 means no better than predicting the average matrix.
- The script prints the RMSE of predicting the mean structure only. Reconstruction errors should be compared with that
  number, and with the size of thermal fluctuations in your system.

## 6. Reading the results

- **Autoencoder against PCA.** If PCA reaches the same error, the data is mostly linear and PCA is enough. A clear
  advantage for the autoencoder at small *L* means there is nonlinear structure.
- **Where the curve flattens.** The error against bottleneck size shows roughly how many degrees of freedom the data
  has. Beyond that point, a larger bottleneck gains little.
- **All against held-out.** A large gap means the model memorized its training sequences. Similar numbers mean
  the compression generalizes to sequences it has not seen.
- **Error over time.** Compare groups A and B and look for rises. Group B stops at unfolding, so its last frames may
  already be unfolded. Time bins contain different numbers of runs, because B runs end early.
- **Error map and per-condition table.** They show which residue pairs and which conditions lose the most information.

## 7. Settings that matter

| setting | effect |
|---|---|
| `Stride` | Time resolution of the data used. Lower means more frames, more disk and slower runs. |
| `Latents` | Bottleneck sizes to sweep. `MAP_Latent` must be one of them. |
| `Epoches`, `Max_train_frame`, `Batch` | Training time is roughly `Epoches × Max_train_frame` frame passes per model, times the number of bottleneck sizes. Time one epoch first. |
| `Holdout` | Fraction of sequences excluded from training. |
| `EVAL_subset` | Test every n-th frame to speed up the testing pass. |
| `Include_C` | Include runs with `unf_time` = 0. |
| `Store_DTYPE` | Precision of the cache. |

## 8. Limitations

- Low reconstruction error shows the geometry can be compressed. It does not show that the latent code keeps what matters
  for predicting `unf_time`. That needs a separate test, such as a regressor trained on the latent vectors.
- The `all` and `train seqs` numbers include frames from training sequences that were not among the `Max_train_frame`
  frames used to fit the model. Only the `held-out seqs` number is fully independent of training.
- Runs have different lengths (group B stops at unfolding), so averages over time mix different sets of runs.
- Results are for one random split of the sequences (`SEED`). A different seed can change the held-out numbers.
- PCA is fitted on a sample of `PCA_frames` frames, not on the full training set.
