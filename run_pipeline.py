"""
Building Age Classification: Madrid -> Amsterdam Transfer
============================================================
Final submission pipeline. Runs end-to-end with a fixed random seed and is
parameterized by `n` (labeled Amsterdam samples per class), per the
submission requirement that sample size be varyable.

Winning configuration (see README.md and SKILL.md for the tuning history):
  Stage 1: Ordinal classifier chain (Frank & Hall decomposition), where each
           binary sub-model is a HistGradientBoostingClassifier wrapped in
           band dropout (dropout_p=0.35) - trained once on all of Madrid.
  Stage 2: Self-training - pools the n true Amsterdam labels/class with
           Stage 1's high-confidence pseudo-labels (confidence_threshold=0.85,
           max_pseudo_per_class=150) on the rest of Amsterdam, then fits a
           shallow RandomForestClassifier.

Usage:
    python run_pipeline.py --madrid /path/to/madrid_train.parquet \
                            --amsterdam /path/to/amsterdam_data.parquet \
                            --n 5 25 50 100 200 \
                            --seed 42 \
                            --out results.json

Requires: pandas, numpy, scikit-learn, pyarrow (for parquet), python-pptx
(only if --build-deliverables is passed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from scripts.preprocess import (
    assign_age_class,
    compute_spectral_indices,
    fill_time_series_gaps,
    make_flat_view,
)
from scripts.build_pixel_features import build_pixel_features
from scripts.feature_engineering_extra_fast import build_extended_features_fast, PID
from scripts.train_stage1 import BandDropoutWrapper
from scripts.ordinal_classifier import OrdinalClassifier
from scripts.transfer_stage2 import self_training, _class_prototypes

RANDOM_STATE = 42  # fixed seed, per submission requirements
BASE_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def load_and_preprocess(path: str, city_label: str) -> pd.DataFrame:
    """Notebook-faithful preprocessing: flatten -> gap-fill -> spectral indices."""
    log(f"loading {city_label} from {path}")
    df = pd.read_parquet(path)
    df = assign_age_class(df)
    flat = make_flat_view(df)
    flat = fill_time_series_gaps(flat)
    flat = compute_spectral_indices(flat)
    log(f"{city_label} preprocessed: {flat.shape}")
    return flat


def build_features(flat: pd.DataFrame) -> pd.DataFrame:
    """Baseline 60-feature set + extended temporal features, merged on pixel id."""
    base = build_pixel_features(flat)
    ext = build_extended_features_fast(flat)
    return base.merge(ext, on=PID, how="left")


def make_stage1_model():
    """The winning Stage 1 architecture: ordinal chain + band-dropout HGB."""
    def make_bd_hgb(band_groups):
        return BandDropoutWrapper(
            HistGradientBoostingClassifier(
                max_depth=6, learning_rate=0.06, max_iter=200, random_state=0
            ),
            band_groups, dropout_p=0.35, seed=RANDOM_STATE,
        )
    return make_bd_hgb  # factory needs band_groups bound at call time; see main()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--madrid", required=True, help="Path to madrid_train.parquet")
    parser.add_argument("--amsterdam", required=True, help="Path to amsterdam_data.parquet")
    parser.add_argument("--n", type=int, nargs="+", default=[5, 25, 50, 100, 200],
                         help="Amsterdam labeled samples per class to evaluate (default: submission's required 5 values)")
    parser.add_argument("--n-trials", type=int, default=15, help="Repeated random support-set draws per n")
    parser.add_argument("--n-folds", type=int, default=5, help="Madrid CV folds")
    parser.add_argument("--n-repeats", type=int, default=2, help="Madrid CV repeats")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", default="results.json", help="Where to write the results JSON")
    parser.add_argument("--build-deliverables", action="store_true",
                         help="Also build submission.pptx and justification.md from the results")
    args = parser.parse_args()

    t0 = time.time()
    np.random.seed(args.seed)

    # --- 1. Preprocess both cities (notebook-faithful) ---
    flat_madrid = load_and_preprocess(args.madrid, "Madrid")
    flat_amsterdam = load_and_preprocess(args.amsterdam, "Amsterdam")

    # --- 2. Build features (baseline + extended) ---
    log("building features...")
    feat_madrid = build_features(flat_madrid)
    feat_amsterdam = build_features(flat_amsterdam)
    drop_cols = set(PID + ["age_class", "weighted_mean_year"])
    feature_cols = [c for c in feat_madrid.columns if c not in drop_cols and c in feat_amsterdam.columns]
    log(f"{len(feature_cols)} features")

    X_madrid = feat_madrid[feature_cols].values.astype(np.float64)
    y_madrid = feat_madrid["age_class"].values.astype(int)
    X_amsterdam = feat_amsterdam[feature_cols].values.astype(np.float64)
    y_amsterdam = feat_amsterdam["age_class"].values.astype(int)

    scaler = StandardScaler().fit(X_madrid)  # fit on Madrid only - no Amsterdam leakage
    X_madrid_s = scaler.transform(X_madrid)
    X_amsterdam_s = scaler.transform(X_amsterdam)

    band_groups = {b: [i for i, c in enumerate(feature_cols) if c.startswith(b)] for b in BASE_BANDS}

    # --- 3. Stage 1: Madrid cross-validation (mean/std macro F1 + confusion matrix) ---
    log("Stage 1: cross-validating on Madrid...")
    rskf = RepeatedStratifiedKFold(n_splits=args.n_folds, n_repeats=args.n_repeats, random_state=args.seed)
    make_bd_hgb = make_stage1_model()
    fold_f1, oof_true, oof_pred = [], [], []
    for tr_idx, val_idx in rskf.split(X_madrid_s, y_madrid):
        model = OrdinalClassifier(lambda bg=band_groups: make_bd_hgb(bg))
        model.fit(X_madrid_s[tr_idx], y_madrid[tr_idx])
        pred = model.predict(X_madrid_s[val_idx])
        fold_f1.append(f1_score(y_madrid[val_idx], pred, average="macro"))
        oof_true.extend(y_madrid[val_idx].tolist())
        oof_pred.extend(pred.tolist())
    madrid_cv = {"mean_f1": float(np.mean(fold_f1)), "std_f1": float(np.std(fold_f1))}
    classes = sorted(np.unique(y_madrid))
    cm = confusion_matrix(oof_true, oof_pred, labels=classes)
    madrid_confusion = (cm / cm.sum(axis=1, keepdims=True)).tolist()
    log(f"Madrid CV macro F1: {madrid_cv['mean_f1']:.4f} +/- {madrid_cv['std_f1']:.4f} ({time.time()-t0:.0f}s elapsed)")

    # --- 4. Fit final Stage 1 model on ALL of Madrid ---
    log("fitting final Stage 1 model on full Madrid...")
    stage1_model = OrdinalClassifier(lambda bg=band_groups: make_bd_hgb(bg))
    stage1_model.fit(X_madrid_s, y_madrid)
    zero_shot_f1 = f1_score(y_amsterdam, stage1_model.predict(X_amsterdam_s), average="macro")
    log(f"zero-shot Amsterdam F1: {zero_shot_f1:.4f}")

    # precompute Stage 1 inference on all of Amsterdam once (self_training reuses this per trial)
    full_proba = stage1_model.predict_proba(X_amsterdam_s)
    full_classes = stage1_model.classes_
    madrid_protos = _class_prototypes(X_madrid_s, y_madrid)

    # --- 5. Stage 2: few-shot transfer sweep over the required n values ---
    log(f"Stage 2: self-training sweep over n={args.n}...")
    amsterdam_results = {}
    classes_ams = np.unique(y_amsterdam)
    for n in args.n:
        rng = np.random.default_rng(args.seed + n)
        scores = []
        for _ in range(args.n_trials):
            support_idx = []
            for c in classes_ams:
                c_idx = np.where(y_amsterdam == c)[0]
                support_idx.extend(rng.choice(c_idx, min(n, len(c_idx)), replace=False).tolist())
            support_idx = np.array(support_idx)
            query_mask = np.ones(len(y_amsterdam), dtype=bool)
            query_mask[support_idx] = False
            X_sup, y_sup = X_amsterdam_s[support_idx], y_amsterdam[support_idx]
            X_qry, y_qry = X_amsterdam_s[query_mask], y_amsterdam[query_mask]

            pred = self_training(
                X_sup, y_sup, X_qry,
                madrid_prototypes=madrid_protos,
                precomputed_proba=full_proba[query_mask], precomputed_classes=full_classes,
                confidence_threshold=0.85, max_pseudo_per_class=150,
            )
            scores.append(f1_score(y_qry, pred, average="macro", zero_division=0))

        amsterdam_results[n] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        log(f"  n={n}: F1={amsterdam_results[n]['mean']:.4f} +/- {amsterdam_results[n]['std']:.4f} ({time.time()-t0:.0f}s elapsed)")

    # --- 6. Save results ---
    results = {
        "madrid_cv": madrid_cv,
        "madrid_confusion_matrix": madrid_confusion,
        "madrid_confusion_labels": classes,
        "zero_shot_amsterdam_f1": zero_shot_f1,
        "amsterdam_results": amsterdam_results,
        "avg_f1": float(np.mean([v["mean"] for v in amsterdam_results.values()])),
        "peak_f1": max(v["mean"] for v in amsterdam_results.values()),
        "config": {
            "seed": args.seed, "n_trials": args.n_trials, "n_folds": args.n_folds,
            "n_repeats": args.n_repeats, "n_features": len(feature_cols),
            "stage1": "OrdinalClassifier(BandDropoutWrapper(HGB, dropout_p=0.35))",
            "stage2": "self_training(confidence_threshold=0.85, max_pseudo_per_class=150)",
        },
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"results written to {args.out}")
    log(f"avg F1={results['avg_f1']:.4f}  peak F1={results['peak_f1']:.4f}  total time={time.time()-t0:.0f}s")

    # --- 7. Optional: build PPTX + justification deliverables ---
    if args.build_deliverables:
        from scripts.build_deliverables import build_pptx
        design_decisions = [
            "Stage 1: ordinal classifier chain + band dropout (rate 0.35) - improves Madrid CV AND zero-shot Amsterdam F1 simultaneously",
            "Band dropout rate and self-training confidence threshold/pseudo-label cap were grid-searched (see SKILL.md)",
            "Stage 2: self-training beats plain and shrinkage prototypes at every tested budget",
        ]
        build_pptx(
            "submission.pptx",
            abstract="[fill in - see written_justification_tuned.md draft for language to adapt]",
            design_decisions=design_decisions,
            madrid_cv=madrid_cv,
            amsterdam_results=amsterdam_results,
        )
        log("submission.pptx built (fill in the abstract placeholder before submitting)")


if __name__ == "__main__":
    main()
