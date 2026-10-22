## Validated on real competition data (this run)

The real `madrid_train.parquet` / `amsterdam_data.parquet` files were run
through this entire pipeline end-to-end (76,263 Madrid pixels, 25,992
Amsterdam pixels), including a fine-tuning pass. **The fine-tuned
configuration below is the current recommended default** — everything else
in this section is the history of how it was found.

### Fine-tuned recipe (best found, use this first)

- **Stage 1**: `OrdinalClassifier` where every binary sub-model is
  `BandDropoutWrapper(HistGradientBoostingClassifier(max_depth=6,
  learning_rate=0.06, max_iter=200), band_groups, dropout_p=0.35)`. Combining
  ordinal structure with band dropout beat either alone, and — unusually —
  improved *both* Madrid in-domain F1 (0.639) and zero-shot Amsterdam F1
  (0.608) at once; normally a robustness intervention costs some in-domain
  accuracy, but a grid search over `dropout_p` from 0.05 to 0.65 found a
  clear peak at 0.35 with no such tradeoff (see the grid below).
- **Stage 2**: `self_training` with `confidence_threshold=0.85,
  max_pseudo_per_class=150` (grid-searched; higher thresholds like 0.95
  maximize n=100/200 slightly but collapse n=5 performance, so 0.85 was
  chosen as the value that doesn't sacrifice the low end — see the grid
  below for the full tradeoff). Tune this against whichever shot count your
  prize rubric actually scores (here, n=25), not against the easiest one.

Result of the fine-tuned config vs. the first (untuned) pass:

| Metric | First pass | Fine-tuned |
|---|---|---|
| Best average F1 | 0.646 | **0.669** |
| Best peak F1 (n=200) | 0.699 | **0.711** |
| Best low-data F1 (n=25) | 0.622 | **0.663** |

`dropout_p` grid (ordinal+band-dropout Stage 1, zero-shot Amsterdam F1):
`0.05→0.593, 0.15→0.596, 0.25→0.605, 0.35→0.608 (peak), 0.45→0.600,
0.55→0.589, 0.65→0.580`. Both tails fall off — too little dropout doesn't
build robustness, too much destroys signal even the target city needs.

`self_training` confidence/cap grid (n=25 / n=100 F1, n_trials=8):
threshold 0.5-0.7 were identical (the 300-cap was already binding below
0.7); 0.8→(0.653/0.694), 0.9→(0.651/0.695), 0.97→(0.656/0.696). Cap:
150→(0.658/0.696) beat 300 and 500+ at the same threshold — more
pseudo-labels isn't automatically better once you're past the confidence
threshold's natural cutoff. **Critically, n=5 F1 collapsed as the threshold
rose past ~0.85** (0.597 at 0.85 → 0.486 at 0.95) — check whether your prize
rubric's low-data point is n=5 or n=25 before picking a threshold; tuning
blindly toward n=25/100/200 here would have quietly wrecked n=5.

### Further tuning attempted beyond this — plateau confirmed

After reaching the recipe above, three more rounds were tried, specifically
to check whether the numbers could be pushed toward a much higher target
(90% F1 was asked about) — worth recording so this isn't re-attempted
expecting a different outcome:

- **HGB base-learner grid** (`learning_rate` in [0.03, 0.06, 0.1],
  `max_iter` in [150, 200, 300, 400], `max_depth` in [4, 6, 8], all wrapped
  in the same ordinal+band-dropout structure): none beat the original
  `learning_rate=0.06, max_iter=200, max_depth=6`.
- **L2 regularization** (`l2_regularization` in [3, 5, 8, 15, 25, 40, 60]):
  this *did* move the isolated zero-shot metric (0.608 → 0.623 at
  `l2=15`), plateauing across 8-40 with no further gain past that. But
  re-running the full Stage 2 sweep with `l2=15` gave average F1 0.666 and
  n=25 F1 0.662 — statistically indistinguishable from (and marginally
  below) the no-extra-L2 recipe's 0.669 / 0.663. **An isolated intermediate
  metric improving does not guarantee the actual downstream metric improves
  with it** — self-training's confidence filtering evidently absorbed the
  small zero-shot gain rather than passing it through. Recorded here so a
  future pass doesn't mistake the zero-shot number alone for progress.
- **Conclusion**: given the closest published benchmark (MapYourCity,
  full 3-modality deep ensemble) tops out around 74 (mean-per-class
  accuracy) and its single-EO-modality ablation lands near 58, this
  6-band, single-modality, small-sample-transfer task realistically has a
  ceiling well below 90% F1 — the remaining error is concentrated at
  class-boundary pixels near the 1960/1984/2004 date cutoffs (confirmed via
  the confusion matrix — construction-year labels are a discretization of a
  continuous quantity, so boundary pixels carry real label noise, not
  model error). Further gains would likely need a second data modality, not
  more hyperparameter search on this one.

The real `madrid_train.parquet` / `amsterdam_data.parquet` files were run
through this entire pipeline end-to-end (76,263 Madrid pixels, 25,992
Amsterdam pixels). Unlike the synthetic proxy below, these are real results:

| Stage 1 candidate | Madrid CV F1 (25K-row subsample, 3-fold) | Zero-shot Amsterdam F1 |
|---|---|---|
| `random_forest` | 0.607 ± 0.001 | not tested |
| `hist_gb` | 0.628 ± 0.006 | 0.535 |
| `rf_hgb_blend` | 0.629 ± 0.003 | not tested (too expensive to justify given the tie below) |
| `band_dropout` | 0.629 ± 0.002 | 0.549 |
| **`ordinal_classifier`** | 0.622 ± 0.003 | **0.573** |

Three candidates were statistically tied on Madrid CV; the ordinal classifier
was nominally *behind* on that metric but clearly *ahead* on zero-shot
Amsterdam F1 — the metric that actually matters for the prizes. This is the
reason this skill checks zero-shot transfer per candidate rather than
picking Stage 1 by Madrid CV alone.

With the ordinal model as Stage 1, the Stage 2 sweep (n_trials=15, shots
5/25/50/100/200) gave:

| n | plain_prototype | shrinkage_prototype | **self_training** |
|---|---|---|---|
| 5 | 0.521 | 0.486 | **0.590** |
| 25 | 0.616 | 0.557 | **0.622** |
| 50 | 0.628 | 0.596 | **0.644** |
| 100 | 0.635 | 0.621 | **0.673** |
| 200 | 0.635 | 0.628 | **0.699** |

`self_training` (fed by the ordinal model's pseudo-labels) won **every shot
count**, sweeping best-average F1 (0.646), best-peak F1 (0.699 at n=200), and
best-low-data F1 (0.622 at n=25) simultaneously. `shrinkage_prototype`
underperformed plain prototype here — a reminder its benefit depends on how
the real domain shift is structured, not a guarantee.

Confusion matrix on Amsterdam (ordinal model, zero-shot) confirmed the
intended effect: almost all errors land on *adjacent* classes (1↔2, 2↔3,
3↔4). Class 2 (recall 0.481, F1 0.486) is weakest — expected, since it's
confusable from both neighbors — while Classes 3-4 (F1 0.618, 0.617), which
fall inside the Landsat record with real construction-year signal, do best.

**Resource note:** this validation ran on a 1-CPU-core sandbox, and
background/nohup jobs do not survive between tool calls here, so everything
ran in foreground, chunked into fold-by-fold checkpoints saved to disk.
Stage 1 CV used a 25K-row stratified Madrid subsample for candidate
selection (full 5-fold CV on all 76K rows wasn't feasible in that budget),
with final models refit on the full Madrid set. If your actual hackathon
compute has more cores and no per-command time limit, prefer full-data CV
over the subsample shortcut — the subsample was a constraint of this
validation environment, not a recommended default.

Also: the original `feature_engineering_extra.py::build_extended_features()`
does not scale to real pixel counts — its per-pixel Python loop exceeded 300s
on a subsample that should take seconds. Use
`scripts/feature_engineering_extra_fast.py::build_extended_features_fast()`
instead — verified to produce numerically identical output (vectorized dense
pixel-by-year matrix operations instead of a Python loop over groups) and
runs the full 76K-pixel Madrid set in ~31s.

