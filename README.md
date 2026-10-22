# Building Age Classification: Final Submission Code

Madrid -> Amsterdam transfer for building construction-era classification
from Landsat spectral time series. This is the winning configuration from
the full tuning process (see `SKILL_NOTES.md` for the grid searches and
reasoning behind every choice below).

## Winning configuration

- **Stage 1 (Madrid)**: an ordinal classifier chain (Frank & Hall
  decomposition) — the classes are temporally ordered, so this respects
  that structure instead of treating them as unrelated categories. Each of
  the chain's binary sub-models is a `HistGradientBoostingClassifier`
  wrapped in band dropout (`dropout_p=0.35`): randomly zeroing a whole
  spectral band's features per training row so the model can't over-rely
  on any single band whose calibration might not hold in Amsterdam.
- **Stage 2 (Amsterdam transfer)**: self-training. Pools the `n` true
  Amsterdam labels/class with Stage 1's high-confidence pseudo-labels
  (`confidence_threshold=0.85`, capped at `max_pseudo_per_class=150`) on
  the rest of Amsterdam, then fits a shallow `RandomForestClassifier`.

**Results** (76,263 Madrid pixels / 25,992 Amsterdam pixels, real
competition data): Madrid CV macro F1 ≈0.64, zero-shot Amsterdam F1 ≈0.61-0.62,
few-shot F1 ranging 0.58 (n=5) to 0.71 (n=200). Average F1 0.669, peak F1
0.711, low-data (n=25) F1 0.663.

## Requirements

```
pip install pandas numpy scikit-learn pyarrow python-pptx
```

## Running it

```bash
python run_pipeline.py \
    --madrid /path/to/madrid_train.parquet \
    --amsterdam /path/to/amsterdam_data.parquet \
    --n 5 25 50 100 200 \
    --seed 42 \
    --out results.json \
    --build-deliverables
```

- `--n` is the list of Amsterdam labeled-samples-per-class budgets to
  evaluate — the submission spec requires 5, 25, 50, 100, 200 (the
  default), but the script accepts any list, so you can vary it
  (e.g. `--n 25` to test a single budget quickly).
- `--seed` fixes all randomness (CV splits, support-set draws) for
  reproducibility, per the submission requirement.
- `--build-deliverables` additionally writes `submission.pptx` (abstract
  placeholder — fill it in, see `written_justification_tuned.md` for
  language to adapt) using the actual numbers this run produced.
- Runtime: on a single CPU core, expect roughly 5-8 minutes for
  preprocessing both cities, a few more minutes for Madrid CV
  (`--n-folds 5 --n-repeats 2` by default), and a couple minutes for the
  Stage 2 sweep. Reduce `--n-folds`/`--n-repeats`/`--n-trials` for a
  faster, noisier run.

## File layout

```
run_pipeline.py              <- entry point (this is requirement #2, "training code")
scripts/
  preprocess.py               notebook-faithful flatten -> gap-fill -> spectral indices
  build_pixel_features.py     baseline 60-feature set (mirrors Notebook 3)
  feature_engineering_extra_fast.py   extended temporal features, vectorized for real-data scale
  ordinal_classifier.py       Frank & Hall ordinal classifier chain
  train_stage1.py             Stage 1 candidates, incl. BandDropoutWrapper, fold ensembling
  transfer_stage2.py          Stage 2 strategies (plain/shrinkage prototype, self-training)
  build_deliverables.py       PPTX + word-limit-checked justification builder
```

`train_stage1.py` and `transfer_stage2.py` also contain the *other*
candidates that were compared during tuning (plain Random Forest, HGB
alone, plain ordinal, plain band-dropout, prototype-based transfer) — kept
in for transparency and in case you want to re-run the comparison on the
real held-out test sets, not because they're all used by `run_pipeline.py`
directly (which calls the winning combination only).

## What's *not* included here

- The synthetic validation dataset generator (`synthetic_validation_data.py`)
  used before the real data was available — no longer needed once you have
  the real parquet files.
- The full tuning history / grid search results — see `SKILL_NOTES.md`.
