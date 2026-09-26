# Running pipeline steps on Kaggle

`kjob.py` pushes any `python -m matching.<step>` command to a private Kaggle
script kernel, follows its log, and downloads the results. The code in `src/`
is embedded in the kernel on every push, so there is no code dataset to keep in sync.

Kaggle CPU sessions have 4 cores, 31 GB of RAM and about 1 TB of `/tmp`. That is
fewer cores than the laptop but about 9x the free RAM. Runs are capped at 12 h,
and `/kaggle/working` (the downloadable output) is capped at 20 GB.

## One-time setup

```bash
pip install kaggle
# kaggle.com -> Settings -> API -> Create New Token, then move kaggle.json to ~/.kaggle/
```

## Workflow

```bash
cd kaggle
export PYTHONUTF8=1

# 1. Upload inputs as a private dataset. The globs are relative to ER_ROOT.
#    Re-running with the same name uploads a new version.
python kjob.py data er-full-train "work/match/full_train/features_*.parquet" work/match/sample10/s1.parquet

# 2. Run. Each --cmd is "<module> <args>" and the commands run in order.
python kjob.py run er-train-full --data er-full-train --timeout 43200 --wait \
    --cmd "matching.train_full --features full_train/features_*.parquet --tag full"

# 3. Download the outputs to $ER_ROOT/kaggle/runs/<slug>/ with their normal layout.
python kjob.py fetch er-train-full
```

Use `status <slug>` and `wait <slug>` to check on or follow a run started earlier.
Kaggle only serves a run's log after its script has finished, so `wait` shows
queued/running status until then and prints the whole log at the end.
Use `--kernel <slug>` to attach an earlier run's outputs as inputs, for example a
trained model for `matching.predict`.

## Details

- **Input paths.** Paths are flattened with `__` for `/`. For example,
  `work/match/sample10/s1.parquet` is uploaded as
  `work__match__sample10__s1.parquet`. On Kaggle it is linked back to the same
  path under a stand-in `ER_ROOT` (`/tmp/er`), so the commands run unchanged.
- **What gets downloaded.** Only new files under `work/` and `output/` that
  match `*.txt *.json *.npy *.tsv *.log *.csv`, plus any `--keep` globs, and
  each at most `--max-out-mb`. Intermediate Parquet files such as `trainset.parquet`
  stay on Kaggle.
- **Threads.** LightGBM's `num_threads` (hard-coded to 12 for the laptop) is set to
  Kaggle's core count at run time. The source files are not changed.
- **Missing packages.** `rapidfuzz` and `Unidecode` are not on the Kaggle image.
  The runner pip-installs them, which needs internet enabled on the kernel
  (the default; the account must be phone-verified).
