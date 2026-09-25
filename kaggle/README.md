# Running TRICE on Kaggle

`trice_kaggle_runner.ipynb` runs the whole pipeline end to end inside a Kaggle notebook and
writes the submittable `matching_results.tsv` to the notebook output.

## One-time setup

1. **New Notebook** on Kaggle (or *File → Import Notebook* and upload
   `trice_kaggle_runner.ipynb`).
2. **Add the dataset.** Right panel → *Add Input* → *Datasets* → your uploaded challenge
   dataset. Upload the seven TSVs keeping the folder structure:

   ```
   train/train_source1.tsv   train/train_source2.tsv   train/train_source3.tsv
   train/train_ground_truth.tsv
   test/test_source1.tsv     test/test_source2.tsv     test/test_source3.tsv
   ```

   The notebook auto-detects the mount under `/kaggle/input/`. If your upload nests the
   folders differently, set `DATA_DIR` by hand in the **configuration** cell.
3. **Settings** (right panel):
   - Accelerator: **None** — the models are gradient-boosted trees; a GPU gives no benefit.
   - Internet: **On** — needed to `git clone` the code and `pip install` two small packages.
4. **Run All.**

## What it does

| cell | stage | output |
|---|---|---|
| clone | pulls `https://github.com/iittjjee2024/TRICE.git` | `/kaggle/working/TRICE` |
| deps | installs `rapidfuzz`, `Unidecode` (rest ship with Kaggle) | — |
| config | auto-detects the dataset, sets a smoke/full switch | — |
| link | points the repo's expected paths at the Kaggle mounts | — |
| tests | `test_decide.py`, `test_union.py` (need no data) | proofs pass |
| 02 | mine token aliases from ground truth | `artifacts/variants.json` |
| 03 | normalise 24 M records to Parquet | `artifacts/store/` |
| 05 | train the matcher + score a held-out split | `artifacts/runs/kaggle/` |
| 07 | tune the decision layer on validation | updates the bundle |
| 06 | full test inference | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| validate | runs the official validator, prints a head sample | `PASS` |
| export | copies the TSVs to `/kaggle/working` for download | download from *Output* |

## Fast smoke run first

The config cell defaults to `SUBSET = 4000` (train entities per country) for a quick
end-to-end check. Set `SUBSET = None` for the real submission run. Full inference over the
1.73 M test entities takes roughly 1.5–2.5 h on Kaggle CPU.

## Getting the result out

After a full run, open the notebook's **Output** tab and download
`matching_results.tsv` — that is the file you upload to the challenge portal. To turn the
predictions into a Kaggle Dataset (e.g. to chain notebooks), *Save Version* with the files
under `/kaggle/working`.

## Memory

Kaggle notebooks provide ~30 GB RAM, comfortably above the 16 GB the pipeline was tuned
for, so the defaults are safe. If you do hit a limit during inference, lower
`--query-batch` (e.g. to `60000`) in the stage-5 cell.
