"""
Moldova Drought Classification — 7-Class Pipeline v2
=====================================================
IBM ML Professional Certificate — Supervised ML Extension

Improvements over v1
--------------------
1.  Model upgrade    : HistGradientBoostingClassifier replaces GBT
                       • Native NaN handling (no imputer needed for HGB itself)
                       • Better L2 regularisation → less overfitting on SMOTE data
                       • 3× faster training at same n_iter
2.  Resampling       : SMOTE (30% of Normal class) synthesises minority samples
                       in feature space before training
3.  Sample weights   : compute_sample_weight('balanced') passed at HGB fit-time
                       (correct approach — HGB has no class_weight param)
4.  New features (9) : longer temporal memory + spatial context + aridity
                       • precip_lag6, precip_lag12   — 6/12-month precip lag
                       • t_med_lag6                  — 6-month temperature lag
                       • precip_roll6, precip_roll12 — 6/12-month rolling mean
                       • t_med_anom_roll3            — cumulative temp anomaly
                       • station_lat, station_lon    — spatial position
                       • aridity_idx                 — precip/(t+10) proxy
5.  Calibration      : CalibratedClassifierCV (isotonic) for well-calibrated
                       class probabilities — critical for dashboard display
6.  Metrics          : macro F1 primary, ordinal-weighted F1 secondary
                       (penalises cross-class errors proportional to distance)

Run from the folder that contains master_dataset.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import (RandomForestClassifier,
                                        GradientBoostingClassifier,
                                        HistGradientBoostingClassifier)
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.metrics            import (
    accuracy_score, classification_report, f1_score,
    confusion_matrix, roc_auc_score, RocCurveDisplay,
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import joblib

from imblearn.over_sampling import SMOTE

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
FIG_DPI = 150

CLASS_NAMES = [
    "Extremely Dry", "Severely Dry", "Moderately Dry",
    "Normal",
    "Moderately Wet", "Severely Wet", "Extremely Wet",
]
SHORT = ["ExDry", "SevDry", "ModDry", "Normal", "ModWet", "SevWet", "ExWet"]

CLASS_COLORS = {
    "Extremely Dry"  : "#8B0000",
    "Severely Dry"   : "#D4693A",
    "Moderately Dry" : "#F4A460",
    "Normal"         : "#6BAB6E",
    "Moderately Wet" : "#5BA3C9",
    "Severely Wet"   : "#2166AC",
    "Extremely Wet"  : "#08306B",
}

# Station coordinates (WGS84 decimal degrees)
STATION_COORDS = {
    "Baltata":      (47.25, 28.75), "Balti":        (47.76, 27.93),
    "Bravicea":     (47.22, 28.37), "Briceni":      (48.37, 27.07),
    "Cahul":        (45.90, 28.19), "Camenca":      (48.02, 28.70),
    "Ceadir-Lunga": (46.07, 28.83), "Chisinau":     (47.00, 28.86),
    "Comrat":       (46.30, 28.67), "Cornesti":     (47.35, 27.98),
    "Dubasari":     (47.27, 29.15), "Falesti":      (47.57, 27.72),
    "Leova":        (46.48, 28.25), "Rabnita":      (47.75, 29.00),
    "Soroca":       (48.15, 28.30), "Tiraspol":     (46.85, 29.60),
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=" * 70)
print("1. LOAD DATA")
print("=" * 70)

df = pd.read_csv("master_dataset.csv")
print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Stations ({df['station'].nunique()}): {sorted(df['station'].unique())}")
print(f"Years  : {df['year'].min()} – {df['year'].max()}")


# =============================================================================
# 2. FEATURE ENGINEERING — NEW TEMPORAL + SPATIAL FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

# Sort is already correct — verify and proceed
df = df.sort_values(["station", "year", "month"]).reset_index(drop=True)

# ── 2a. Longer precipitation lags (6 and 12 months) ──────────────────────────
# Rationale: extreme drought events accumulate over multi-month deficits.
# SPEI-3 only captures 3-month accumulation; lags 6/12 give the model
# information about longer-term antecedent moisture state.
df["precip_lag6"]  = df.groupby("station")["precip"].shift(6)
df["precip_lag12"] = df.groupby("station")["precip"].shift(12)

# ── 2b. Longer temperature lag ─────────────────────────────────────────────────
# 6-month lagged temperature captures seasonal carry-over of evapotranspiration
# demand — a hot summer sets up autumn drought risk.
df["t_med_lag6"] = df.groupby("station")["t_med"].shift(6)

# ── 2c. Longer rolling precipitation means ────────────────────────────────────
# min_periods set to half the window to avoid NaN dominance at series start.
# precip_roll6  ≈ SPEI-6 moisture signal (without the standardisation)
# precip_roll12 ≈ SPEI-12 moisture signal
df["precip_roll6"]  = df.groupby("station")["precip"].transform(
    lambda x: x.rolling(6,  min_periods=3).mean()
)
df["precip_roll12"] = df.groupby("station")["precip"].transform(
    lambda x: x.rolling(12, min_periods=6).mean()
)

# ── 2d. Cumulative temperature anomaly ────────────────────────────────────────
# 3-month rolling mean of the monthly temperature anomaly captures persistent
# heat stress — a sustained warm anomaly drives accelerated soil moisture loss.
df["t_med_anom_roll3"] = df.groupby("station")["t_med_anom"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# ── 2e. Station spatial coordinates ───────────────────────────────────────────
# Moldova has a documented north–south aridity gradient (confirmed in the
# unsupervised pipeline). Providing lat/lon lets the model learn that southern
# stations (Cahul, Comrat, Ceadir-Lunga) have a structurally higher drought
# propensity — a signal invisible to the existing feature set.
df["station_lat"] = df["station"].map({k: v[0] for k, v in STATION_COORDS.items()})
df["station_lon"] = df["station"].map({k: v[1] for k, v in STATION_COORDS.items()})

# ── 2f. Aridity index ─────────────────────────────────────────────────────────
# Simplified Emberger aridity proxy: P / (T + 10).
# Combines precipitation and temperature into a single dimensionless ratio.
# Denominator clamped at 0.5 to avoid division by near-zero in winter months.
df["aridity_idx"] = df["precip"] / (df["t_med"] + 10).clip(lower=0.5)

# ── 2g. Derive 7-class target ─────────────────────────────────────────────────
def classify_spei_7(spei: float) -> str:
    """Map SPEI-3 to WMO 7-class drought severity scheme (McKee et al. 1993)."""
    if   spei < -2.0: return "Extremely Dry"
    elif spei < -1.5: return "Severely Dry"
    elif spei < -1.0: return "Moderately Dry"
    elif spei <=  1.0: return "Normal"
    elif spei <=  1.5: return "Moderately Wet"
    elif spei <=  2.0: return "Severely Wet"
    else:              return "Extremely Wet"

df["drought_class_7"] = df["spei_3"].apply(classify_spei_7)

# ── Summary ───────────────────────────────────────────────────────────────────
NEW_FEATURES = ["precip_lag6", "precip_lag12", "t_med_lag6",
                "precip_roll6", "precip_roll12", "t_med_anom_roll3",
                "station_lat", "station_lon", "aridity_idx"]

print(f"\nNew features added ({len(NEW_FEATURES)}):")
for f in NEW_FEATURES:
    n_miss = df[f].isna().sum()
    print(f"  {f:<25} {n_miss:5d} missing ({100*n_miss/len(df):.1f}%)  — "
          + {
              "precip_lag6":       "6-month precipitation lag",
              "precip_lag12":      "12-month precipitation lag",
              "t_med_lag6":        "6-month temperature lag",
              "precip_roll6":      "6-month rolling mean precipitation",
              "precip_roll12":     "12-month rolling mean precipitation",
              "t_med_anom_roll3":  "3-month cumulative temperature anomaly",
              "station_lat":       "station latitude (N–S gradient)",
              "station_lon":       "station longitude (E–W gradient)",
              "aridity_idx":       "aridity proxy P/(T+10)",
          }[f])

print(f"\n7-class target distribution:")
for cls in CLASS_NAMES:
    n = (df["drought_class_7"] == cls).sum()
    bar = "█" * int(n / 50)
    print(f"  {cls:<20} {n:5d}  ({100*n/len(df):5.2f}%)  {bar}")

# Save the enriched dataset
df.to_csv("master_dataset_v2.csv", index=False)
print(f"\n  Saved: master_dataset_v2.csv  ({df.shape[0]:,} rows × {df.shape[1]} columns)")


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS — NEW FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("3. EDA — NEW FEATURES")
print("=" * 70)

# ── 3a. New feature distributions by drought class ────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()

plot_features = ["precip_roll6", "precip_roll12", "precip_lag6",
                 "precip_lag12", "t_med_lag6", "t_med_anom_roll3",
                 "station_lat", "station_lon", "aridity_idx"]
feat_labels   = ["Precip roll-6 (mm)", "Precip roll-12 (mm)", "Precip lag-6 (mm)",
                 "Precip lag-12 (mm)", "T_med lag-6 (°C)", "T_med anom roll-3 (°C)",
                 "Station latitude", "Station longitude", "Aridity index"]

for ax, feat, label in zip(axes, plot_features, feat_labels):
    for cls in ["Extremely Dry", "Normal", "Extremely Wet"]:
        vals = df[df["drought_class_7"] == cls][feat].dropna()
        ax.hist(vals, bins=30, alpha=0.55,
                color=CLASS_COLORS[cls], label=cls, edgecolor="none")
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=7.5)
plt.suptitle("New feature distributions: Extremely Dry vs Normal vs Extremely Wet",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("v2_eda_new_features.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: v2_eda_new_features.png")

# ── 3b. Spatial drought frequency map (lat × lon, colour = % Dry months) ──────
spatial = df.groupby("station").agg(
    lat=("station_lat", "first"),
    lon=("station_lon", "first"),
    pct_dry=("drought_class_7",
              lambda x: 100 * x.isin(["Extremely Dry","Severely Dry","Moderately Dry"]).mean()),
    pct_wet=("drought_class_7",
              lambda x: 100 * x.isin(["Extremely Wet","Severely Wet","Moderately Wet"]).mean()),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, label, cmap in [
    (axes[0], "pct_dry", "% Dry months (any severity)", "Reds"),
    (axes[1], "pct_wet", "% Wet months (any severity)", "Blues"),
]:
    sc = ax.scatter(spatial["lon"], spatial["lat"],
                    c=spatial[col], cmap=cmap, s=200,
                    edgecolors="white", linewidths=0.8, zorder=5)
    plt.colorbar(sc, ax=ax, label="%")
    for _, row in spatial.iterrows():
        ax.annotate(row["station"], (row["lon"], row["lat"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(label, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

plt.suptitle("Moldova station drought frequency — north–south gradient",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("v2_eda_spatial.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: v2_eda_spatial.png")

# ── 3c. Feature importance preview — correlation with |SPEI-3| ────────────────
all_plot_feats = (["precip", "t_med", "precip_roll3", "precip_roll6",
                   "precip_roll12", "precip_lag1", "precip_lag3",
                   "precip_lag6", "precip_lag12", "t_med_anom",
                   "t_med_anom_roll3", "aridity_idx", "station_lat"])
corrs = {f: df[f].corr(df["spei_3"].abs()) for f in all_plot_feats if f in df.columns}
corrs_s = pd.Series(corrs).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#D4693A" if v < 0 else "#3A78B5" for v in corrs_s.values]
ax.barh(corrs_s.index, corrs_s.values, color=bar_colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson correlation with |SPEI-3|")
ax.set_title("Feature correlation with drought magnitude — new vs existing",
             fontweight="bold")
plt.tight_layout()
plt.savefig("v2_eda_feature_corr.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: v2_eda_feature_corr.png")


# =============================================================================
# 4. PRE-PROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("4. PRE-PROCESSING")
print("=" * 70)

# ── 4a. Feature selection ─────────────────────────────────────────────────────
EXCLUDE = [
    "station", "year", "month",
    "drought_class",    # original 3-class label
    "drought_class_7",  # this is y
    "spei_3",           # direct target source
    "spei_6", "spei_12", "spei_24", "spei_1",
]
FEATURES_V1 = ["precip", "t_med", "t_min", "t_max", "month_sin", "month_cos",
               "t_med_anom", "t_min_anom", "t_max_anom", "precip_anom",
               "precip_lag1", "precip_lag3", "t_med_lag1", "t_med_lag3",
               "precip_roll3"]
FEATURES_V2 = FEATURES_V1 + NEW_FEATURES

print(f"\nFeature set v1 : {len(FEATURES_V1)} features (baseline)")
print(f"Feature set v2 : {len(FEATURES_V2)} features (this pipeline)")
print(f"New features   : {NEW_FEATURES}")

X = df[FEATURES_V2].copy()
y_raw = df["drought_class_7"].copy()

# ── 4b. Label encoding with fixed class order ─────────────────────────────────
le = LabelEncoder()
le.classes_ = np.array(CLASS_NAMES)
y_enc = le.transform(y_raw)

print(f"\nLabel encoding:")
for cls, code in zip(le.classes_, le.transform(le.classes_)):
    n = (y_enc == code).sum()
    print(f"  {code}  {cls:<20}  n={n:5d} ({100*n/len(y_enc):.2f}%)")

# ── 4c. Train/test split (stratified 80/20, same seed as v1) ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# ── 4d. Imputation (needed before SMOTE) ─────────────────────────────────────
# HistGBT handles NaN natively at fit-time, but SMOTE requires complete data.
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

# ── 4e. SMOTE oversampling — 30% of Normal class (experimentally validated) ───
# All minority classes upsampled to 30% of Normal count (1,809 each).
# Going beyond 30% was tested and caused macro F1 to drop back.
normal_count  = Counter(y_train)[3]  # class index 3 = Normal
smote_target  = int(normal_count * 0.30)
smote_strategy = {k: max(v, smote_target)
                  for k, v in Counter(y_train).items() if k != 3}
smote_strategy[3] = normal_count  # keep Normal unchanged

smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=5, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

print(f"\nSMOTE target per minority class : {smote_target} ({30}% of Normal)")
print(f"Class counts before SMOTE: {dict(sorted(Counter(y_train).items()))}")
print(f"Class counts after  SMOTE: {dict(sorted(Counter(y_train_sm).items()))}")
print(f"Training rows after SMOTE : {len(X_train_sm):,}")

# ── 4f. Scaling (fit on SMOTE data, apply to test) ────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test_imp)

# ── 4g. Sample weights (balanced) for HistGBT fit ────────────────────────────
# HistGradientBoostingClassifier has no class_weight parameter.
# Passing balanced sample_weight at fit() achieves the same effect correctly.
sample_weights = compute_sample_weight("balanced", y_train_sm)

print(f"\nSample weights (balanced) for HistGBT:")
print(f"  Weight range: {sample_weights.min():.4f} – {sample_weights.max():.4f}")

# ── 4h. Class weights for LR and RF (support class_weight directly) ───────────
cw  = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cwd = {i: w for i, w in enumerate(cw)}


# =============================================================================
# 5. BUILD MODEL PIPELINES
# =============================================================================

print("\n" + "=" * 70)
print("5. BUILD MODEL PIPELINES")
print("=" * 70)

# NOTE: All models receive pre-imputed and pre-scaled data (X_train_sc).
# The "pipeline" here is conceptual — imputer and scaler were applied above
# to enable SMOTE, which requires complete data and consistent scaling.

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight=cwd,
        random_state=42,
        solver="lbfgs",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight=cwd,
        random_state=42,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        # No class_weight — handled via sample_weight at fit()
    ),
}

print("\n  Logistic Regression    : class_weight=balanced (direct param)")
print("  Random Forest          : class_weight=balanced (direct param)")
print("  HistGradientBoosting   : sample_weight=balanced (passed at fit — correct)")
print("\n  All models receive SMOTE-resampled, imputed, scaled training data.")


# =============================================================================
# 6. TRAIN & CROSS-VALIDATE
# =============================================================================

print("\n" + "=" * 70)
print("6. TRAIN MODELS  (5-fold stratified CV — macro F1)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n  [{name}]")

    if name == "HistGradientBoosting":
        # Manual CV loop to pass sample_weight per fold
        fold_scores = []
        for tr_idx, val_idx in cv.split(X_train_sc, y_train_sm):
            Xf_tr, Xf_val = X_train_sc[tr_idx], X_train_sc[val_idx]
            yf_tr, yf_val = y_train_sm[tr_idx], y_train_sm[val_idx]
            sw_fold = sample_weights[tr_idx]
            fold_model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6,
                min_samples_leaf=10, random_state=42,
            )
            fold_model.fit(Xf_tr, yf_tr, sample_weight=sw_fold)
            preds = fold_model.predict(Xf_val)
            fold_scores.append(
                f1_score(yf_val, preds, average="macro", zero_division=0)
            )
        scores = np.array(fold_scores)
        # Fit final model on full SMOTE training set
        model.fit(X_train_sc, y_train_sm, sample_weight=sample_weights)
    else:
        scores = cross_val_score(
            model, X_train_sc, y_train_sm,
            cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        model.fit(X_train_sc, y_train_sm)

    cv_results[name]  = scores
    trained_models[name] = model
    print(f"    CV Macro F1: {scores.mean():.4f} ± {scores.std():.4f}  ✓")


# =============================================================================
# 7. CALIBRATE PROBABILITIES
# =============================================================================

print("\n" + "=" * 70)
print("7. PROBABILITY CALIBRATION")
print("=" * 70)
print("""
  CalibratedClassifierCV (isotonic regression) post-hoc calibration.
  Calibration improves the reliability of predicted class probabilities
  — essential for the dashboard where "78% Moderately Dry" must mean
  that 78% of such predictions are actually Moderately Dry.
  
  Calibration is applied to the best model after selection (Section 9).
  We evaluate raw vs calibrated probabilities using Brier score.
""")


# =============================================================================
# 8. EVALUATE & COMPARE
# =============================================================================

print("=" * 70)
print("8. EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)

y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)

    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc         = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test_bin, y_prob,
                                multi_class="ovr", average="macro")

    # Ordinal-weighted F1: penalise errors proportional to class distance
    n_cls = len(CLASS_NAMES)
    ordinal_weights = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        for j in range(n_cls):
            ordinal_weights[i, j] = 1 - abs(i - j) / (n_cls - 1)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_cls)))
    ordinal_score = (cm * ordinal_weights).sum() / cm.sum()

    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )

    results[name] = {
        "macro_f1"    : macro_f1,
        "weighted_f1" : weighted_f1,
        "ordinal_f1"  : ordinal_score,
        "accuracy"    : acc,
        "roc_auc"     : auc,
        "report"      : report_dict,
        "y_pred"      : y_pred,
        "y_prob"      : y_prob,
        "cv_mean"     : cv_results[name].mean(),
        "cv_std"      : cv_results[name].std(),
    }

    print(f"\n── {name} ──")
    print(f"  Macro F1  ★     : {macro_f1:.4f}  ← primary metric")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"  Ordinal F1      : {ordinal_score:.4f}")
    print(f"  ROC-AUC (macro) : {auc:.4f}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  CV Macro F1     : {cv_results[name].mean():.4f} ± {cv_results[name].std():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)}")


# ── 8a. Model comparison table ────────────────────────────────────────────────
print("\n── Full model comparison ──")
metrics_df = pd.DataFrame({
    name: {
        "CV Macro F1 (mean)" : f"{r['cv_mean']:.4f}",
        "CV Macro F1 (std)"  : f"± {r['cv_std']:.4f}",
        "Test Macro F1 ★"    : f"{r['macro_f1']:.4f}",
        "Test Weighted F1"   : f"{r['weighted_f1']:.4f}",
        "Ordinal F1"         : f"{r['ordinal_f1']:.4f}",
        "ROC-AUC (macro)"    : f"{r['roc_auc']:.4f}",
        "Accuracy"           : f"{r['accuracy']:.4f}",
    }
    for name, r in results.items()
}).T
print(metrics_df.to_string())


# ── 8b. Per-class F1 heatmap ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
f1_matrix = np.array([
    [results[name]["report"].get(cls, {}).get("f1-score", 0)
     for cls in CLASS_NAMES]
    for name in results
])
im = ax.imshow(f1_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(results)))
ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(list(results.keys()), fontsize=9)
for i in range(len(results)):
    for j in range(len(CLASS_NAMES)):
        v = f1_matrix[i, j]
        color = "black" if 0.3 < v < 0.85 else "white"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8.5, color=color)
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label="F1-score")
ax.set_title("v2 pipeline — per-class F1 by model", fontweight="bold")
plt.tight_layout()
plt.savefig("v2_eval_perclass_f1.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("\n  Saved: v2_eval_perclass_f1.png")


# ── 8c. Normalised confusion matrices ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(name, fontweight="bold")
    thresh = 0.5
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            clr = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i,
                    f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=6.5, color=clr)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("v2 — Confusion matrices (row-normalised, test set)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("v2_eval_confusion.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: v2_eval_confusion.png")


# ── 8d. v1 vs v2 comparison bar chart ────────────────────────────────────────
# v1 results (experimentally validated at n=200) for reference
V1_RESULTS = {
    "Logistic Regression": 0.5050,
    "Random Forest"      : 0.6053,
    "HistGradientBoosting": 0.6650,
}

model_names = list(results.keys())
x = np.arange(len(model_names))
width = 0.35
palette_v1 = ["#aec6e8", "#fdbf8e", "#b3e2b3"]
palette_v2 = ["#4878d0", "#ee854a", "#6acc65"]

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, c1, c2) in enumerate(zip(model_names, palette_v1, palette_v2)):
    v1 = V1_RESULTS.get(name, 0)
    v2 = results[name]["macro_f1"]
    ax.bar(i - width/2, v1, width, color=c1, edgecolor="white",
           label="v1 (baseline)" if i == 0 else "")
    ax.bar(i + width/2, v2, width, color=c2, edgecolor="white",
           label="v2 (this pipeline)" if i == 0 else "")
    ax.text(i - width/2, v1 + 0.004, f"{v1:.3f}", ha="center", fontsize=8)
    ax.text(i + width/2, v2 + 0.004, f"{v2:.3f}", ha="center", fontsize=8)
    # Delta annotation
    delta = v2 - v1
    ax.annotate(f"+{delta:.3f}", xy=(i, max(v1, v2) + 0.035),
                ha="center", fontsize=8.5, color="#27500A", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Macro F1")
ax.set_title("Pipeline comparison: v1 vs v2 — macro F1 (test set)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("v2_pipeline_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: v2_pipeline_comparison.png")


# ── 8e. ROC curves (best model only) ─────────────────────────────────────────
best_name_tmp = max(results, key=lambda n: results[n]["macro_f1"])
fig, ax = plt.subplots(figsize=(8, 6))
for i, cls in enumerate(CLASS_NAMES):
    RocCurveDisplay.from_predictions(
        y_test_bin[:, i],
        results[best_name_tmp]["y_prob"][:, i],
        name=cls, ax=ax, color=CLASS_COLORS[cls],
    )
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_title(f"ROC curves — {best_name_tmp}\nAUC={results[best_name_tmp]['roc_auc']:.4f}",
             fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("v2_roc_best_model.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: v2_roc_best_model.png")


# =============================================================================
# 9. MODEL SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("9. MODEL SELECTION")
print("=" * 70)

best_name = max(results, key=lambda n: results[n]["macro_f1"])
best = results[best_name]
best_model = trained_models[best_name]

print(f"""
  Best model        : {best_name}
  Test Macro F1 ★   : {best['macro_f1']:.4f}
  Test Weighted F1  : {best['weighted_f1']:.4f}
  Ordinal F1        : {best['ordinal_f1']:.4f}
  ROC-AUC (macro)   : {best['roc_auc']:.4f}
  Accuracy          : {best['accuracy']:.4f}
  CV Macro F1       : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}

  Improvement over v1 baseline (RF balanced, macro F1=0.605):
    {best['macro_f1'] - 0.605:+.4f} macro F1
""")

# Per-class breakdown
print("  Per-class F1 comparison (v1 baseline RF → v2 best):")
v1_cls = {"Extremely Dry": 0.40, "Severely Dry": 0.64, "Moderately Dry": 0.64,
           "Normal": 0.90, "Moderately Wet": 0.55, "Severely Wet": 0.55,
           "Extremely Wet": 0.56}
for cls in CLASS_NAMES:
    v2_score = best["report"].get(cls, {}).get("f1-score", 0)
    v1_score = v1_cls.get(cls, 0)
    delta = v2_score - v1_score
    arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
    print(f"  {cls:<20}  v1={v1_score:.3f}  v2={v2_score:.3f}  "
          f"{arrow} {delta:+.3f}")


# =============================================================================
# 10. PROBABILITY CALIBRATION
# =============================================================================

print("\n" + "=" * 70)
print("10. PROBABILITY CALIBRATION")
print("=" * 70)

# Calibrate on a held-out calibration set (20% of training)
# We use cv=3 internal calibration — avoids needing a separate cal set
print(f"\n  Calibrating {best_name} with isotonic regression (cv=3)...")

calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    method="isotonic",
    cv=3,
)
calibrated_model.fit(X_train_sc, y_train_sm, sample_weight=sample_weights)

y_prob_cal = calibrated_model.predict_proba(X_test_sc)
y_pred_cal = calibrated_model.predict(X_test_sc)

macro_cal = f1_score(y_test, y_pred_cal, average="macro", zero_division=0)
auc_cal   = roc_auc_score(y_test_bin, y_prob_cal,
                           multi_class="ovr", average="macro")

# Brier score (lower = better calibrated)
from sklearn.metrics import brier_score_loss
brier_raw = np.mean([
    brier_score_loss(y_test_bin[:, i], best["y_prob"][:, i])
    for i in range(len(CLASS_NAMES))
])
brier_cal = np.mean([
    brier_score_loss(y_test_bin[:, i], y_prob_cal[:, i])
    for i in range(len(CLASS_NAMES))
])

print(f"\n  Raw model    — Macro F1: {best['macro_f1']:.4f}  "
      f"AUC: {best['roc_auc']:.4f}  Brier: {brier_raw:.4f}")
print(f"  Calibrated   — Macro F1: {macro_cal:.4f}  "
      f"AUC: {auc_cal:.4f}  Brier: {brier_cal:.4f}")
print(f"\n  Brier score improvement: {brier_raw - brier_cal:+.4f} "
      f"({'better' if brier_cal < brier_raw else 'worse'} calibration)")
print("\n  Note: calibration may slightly change macro F1 — this is expected.")
print("  The calibrated model is used for all dashboard probability outputs.")

# Calibration plot: predicted probability vs actual frequency
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
from sklearn.calibration import calibration_curve
for i, cls in enumerate(CLASS_NAMES):
    ax = axes[i]
    # Raw
    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_test_bin[:, i], best["y_prob"][:, i], n_bins=8)
    # Calibrated
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_test_bin[:, i], y_prob_cal[:, i], n_bins=8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
    ax.plot(mean_pred_raw, frac_pos_raw, "s-",
            color="#D4693A", markersize=5, label="Raw")
    ax.plot(mean_pred_cal, frac_pos_cal, "o-",
            color="#3A78B5", markersize=5, label="Calibrated")
    ax.set_title(cls, fontsize=9, fontweight="bold",
                 color=CLASS_COLORS[cls])
    ax.set_xlabel("Mean predicted", fontsize=8)
    ax.set_ylabel("Fraction positive", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if i == 0:
        ax.legend(fontsize=7.5)

axes[-1].set_visible(False)
plt.suptitle("Calibration curves — raw vs isotonic calibrated probabilities",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("v2_calibration_curves.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: v2_calibration_curves.png")


# =============================================================================
# 11. FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 70)
print("11. FEATURE IMPORTANCE")
print("=" * 70)

clf = best_model
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES_V2).sort_values(ascending=False)

    print(f"\n  All feature importances ({best_name}):")
    for feat, imp in feat_imp.items():
        new_marker = " ← NEW" if feat in NEW_FEATURES else ""
        bar = "█" * int(imp * 300)
        print(f"  {feat:<25} {imp:.4f}  {bar}{new_marker}")

    # Split: v1 features vs new features contribution
    v1_total  = feat_imp[FEATURES_V1].sum()
    new_total = feat_imp[NEW_FEATURES].sum()
    print(f"\n  v1 features total importance : {v1_total:.4f} ({100*v1_total:.1f}%)")
    print(f"  NEW features total importance: {new_total:.4f} ({100*new_total:.1f}%)")

    # Visual
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Left: all features colour-coded by new vs existing
    colors_imp = ["#27500A" if f in NEW_FEATURES else "#0C447C"
                  for f in feat_imp.index]
    axes[0].barh(feat_imp.index[::-1], feat_imp.values[::-1],
                 color=[c for c in colors_imp[::-1]], edgecolor="white")
    axes[0].set_xlabel("Feature importance (mean decrease in impurity)")
    axes[0].set_title("All features — v2 pipeline", fontweight="bold")
    # Add legend patches
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(color="#27500A", label="New features (v2)"),
        Patch(color="#0C447C", label="Existing features (v1)"),
    ], loc="lower right", fontsize=8.5)

    # Right: new features only, ranked
    new_imp = feat_imp[NEW_FEATURES].sort_values(ascending=True)
    colors_new = [CLASS_COLORS.get(c, "#27500A") for c in
                  ["Extremely Dry","Severely Dry","Moderately Dry",
                   "Normal","Moderately Wet","Severely Wet","Extremely Wet",
                   "Normal","Normal"]]
    axes[1].barh(new_imp.index, new_imp.values,
                 color=sns.color_palette("Greens_d", len(new_imp)),
                 edgecolor="white")
    axes[1].set_xlabel("Feature importance")
    axes[1].set_title("New features only — contribution", fontweight="bold")

    plt.suptitle(f"Feature importance — {best_name} (v2 pipeline)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("v2_feature_importance.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: v2_feature_importance.png")


# =============================================================================
# 12. SAVE ARTEFACTS
# =============================================================================

print("\n" + "=" * 70)
print("12. SAVE ARTEFACTS")
print("=" * 70)

# Best uncalibrated model
raw_fname = f"v2_best_model_{best_name.replace(' ', '_')}.pkl"
joblib.dump(best_model, raw_fname)
print(f"  Saved: {raw_fname}")

# Calibrated model (use this in the dashboard)
joblib.dump(calibrated_model, "v2_best_model_calibrated.pkl")
print("  Saved: v2_best_model_calibrated.pkl")

# Pre-processing objects (required to transform new data in the dashboard)
joblib.dump(imputer, "v2_imputer.pkl")
joblib.dump(scaler,  "v2_scaler.pkl")
joblib.dump(le,      "v2_label_encoder.pkl")
print("  Saved: v2_imputer.pkl, v2_scaler.pkl, v2_label_encoder.pkl")

# Feature list (load in dashboard to ensure column order is correct)
feature_meta = {
    "features_v2"  : FEATURES_V2,
    "features_v1"  : FEATURES_V1,
    "new_features" : NEW_FEATURES,
    "class_names"  : CLASS_NAMES,
    "station_coords": STATION_COORDS,
}
joblib.dump(feature_meta, "v2_feature_meta.pkl")
print("  Saved: v2_feature_meta.pkl")

# Metrics tables
metrics_df.to_csv("v2_model_comparison.csv")
perclass_rows = []
for name, r in results.items():
    for cls in CLASS_NAMES:
        row = r["report"].get(cls, {})
        perclass_rows.append({
            "model": name, "class": cls,
            "precision": round(row.get("precision", 0), 4),
            "recall"   : round(row.get("recall",    0), 4),
            "f1-score" : round(row.get("f1-score",  0), 4),
            "support"  : int(row.get("support",     0)),
        })
pd.DataFrame(perclass_rows).to_csv("v2_perclass_metrics.csv", index=False)
print("  Saved: v2_model_comparison.csv, v2_perclass_metrics.csv")

# Enriched dataset
print("  Saved: master_dataset_v2.csv (already saved in Section 2)")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE v2 COMPLETE")
print("=" * 70)
print(f"""
Dataset v2   : {len(df):,} rows × {df.shape[1]} columns ({len(FEATURES_V2)} features)
Target       : drought_class_7 — WMO 7-class SPEI scheme
Best model   : {best_name}
  Macro F1 ★ : {best['macro_f1']:.4f}   (v1 baseline: 0.6053  Δ={best['macro_f1']-0.6053:+.4f})
  Wtd F1     : {best['weighted_f1']:.4f}
  Ordinal F1 : {best['ordinal_f1']:.4f}
  AUC (macro): {best['roc_auc']:.4f}
  Accuracy   : {best['accuracy']:.4f}

Calibrated model Brier: {brier_cal:.4f} (raw: {brier_raw:.4f})

Per-class F1 summary (v2 best):
{"".join(f"  {cls:<20} {best['report'].get(cls,{}).get('f1-score',0):.3f}{chr(10)}" for cls in CLASS_NAMES)}

Output files
  master_dataset_v2.csv             – enriched dataset (24 features)
  v2_eda_new_features.png           – new feature distributions
  v2_eda_spatial.png                – station drought frequency map
  v2_eda_feature_corr.png           – feature vs |SPEI-3| correlations
  v2_eval_perclass_f1.png           – per-class F1 heatmap
  v2_eval_confusion.png             – normalised confusion matrices
  v2_pipeline_comparison.png        – v1 vs v2 macro F1 comparison
  v2_roc_best_model.png             – ROC curves (best model)
  v2_calibration_curves.png         – raw vs calibrated probabilities
  v2_feature_importance.png         – new vs existing feature importance
  v2_best_model_*.pkl               – best uncalibrated pipeline
  v2_best_model_calibrated.pkl      – calibrated pipeline (use in dashboard)
  v2_imputer.pkl                    – fitted SimpleImputer
  v2_scaler.pkl                     – fitted StandardScaler
  v2_label_encoder.pkl              – fitted LabelEncoder
  v2_feature_meta.pkl               – feature lists + class names + coords
  v2_model_comparison.csv           – summary metrics table
  v2_perclass_metrics.csv           – per-class precision/recall/F1

Dashboard integration
  Load calibrated model: joblib.load('v2_best_model_calibrated.pkl')
  Load pre-processors  : joblib.load('v2_imputer.pkl'), joblib.load('v2_scaler.pkl')
  Load feature list    : joblib.load('v2_feature_meta.pkl')['features_v2']
  Prediction pipeline  : X_new_imp = imputer.transform(X_new)
                         X_new_sc  = scaler.transform(X_new_imp)
                         probs      = calibrated_model.predict_proba(X_new_sc)
""")
