Building Age Classification from Landsat Imagery
Cross-city transfer learning: train on Madrid, adapt to Amsterdam with very few labels.
Submission code for the Ghana Data Science Regional Hackathon 2026 (KNUST). Our team won the Best Average F1 and Best Peak F1 prizes among 5 teams.
Team: [add teammates' names]
The task
Given the spectral reflectance of a 30 m x 30 m Landsat pixel containing buildings (6 bands: Blue, Green, Red, NIR, SWIR1, SWIR2, plus a time series), predict the age class (1-4) of the buildings in that pixel. The goal is a model that works in a new city with very little local labelled data.
Stage 1: train on Madrid.
Stage 2: adapt to Amsterdam using 5, 25, 50, 100 or 200 labelled pixels per class.
Approach
Instead of learning absolute age classes directly, Stage 1 learns an embedding in which relative age is preserved, because relative judgments transfer between cities better than absolute ones.
Features: 176 features built from the spectral time series (gap-filled series, spectral indices, extended temporal features). Scaling uses unlabeled Madrid and Amsterdam statistics jointly (no labels used, so no leakage).
Ordinal-aware metric learning: batch-hard triplet loss with an ordinal margin, plus an auxiliary cross-entropy head, so nearby age classes sit closer together than distant ones.
Domain alignment: a CORAL loss compares Madrid and unlabeled Amsterdam embeddings during training to reduce the gap between the two cities.
Few-shot heads (Stage 2): several light-weight heads are compared at each label budget: prototype, shrinkage prototype, k-NN, logistic regression, ordinal chain on embeddings, transductive prototype, and self-training with pseudo-labels. The best head depends on how many labels are available.
Results
Macro F1, 10-fold repeated cross-validation, seed 42.
Setting
Best head
Macro F1 (mean +/- std)
Madrid (cross-validation)
n/a
0.623 +/- 0.004
Amsterdam, 0 labels (zero-shot)
n/a
0.601
Amsterdam, 5 per class
self-training
0.642 +/- 0.002
Amsterdam, 25 per class
prototype
0.653 +/- 0.012
Amsterdam, 50 per class
k-NN
0.655 +/- 0.010
Amsterdam, 100 per class
k-NN
0.669 +/- 0.008
Amsterdam, 200 per class
k-NN
0.684 +/- 0.005
Full per-head results are in results_metric.json.
Running it
pip install numpy pandas scikit-learn pyarrow torch

python run_metric_learning.py \
    --madrid /path/to/madrid_train.parquet \
    --amsterdam /path/to/amsterdam_data.parquet \
    --n 5 25 50 100 200 \
    --seed 42 \
    --out results_metric.json
The competition data is not included in this repository. It belongs to the organisers.
Repository layout
run_metric_learning.py        final approach (metric learning + CORAL + few-shot heads)
scripts/
  preprocess.py               flatten, gap-fill, spectral indices, age-class labels
  build_pixel_features.py     baseline feature set
  feature_engineering_extra_fast.py   extended temporal features
  metric_learning.py          embedding network, losses, few-shot heads
stage1_metric.pt              trained Stage 1 (Madrid) model
results_metric.json           final cross-validated results
run_pipeline.py               earlier baseline (ordinal gradient boosting + self-training)
run_pipeline.py and its helper scripts are an earlier tree-based baseline kept for comparison. The final results above come from run_metric_learning.py.
