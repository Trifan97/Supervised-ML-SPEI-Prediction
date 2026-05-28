"""
Moldova Drought Classification — 7-Class Modelling Pipeline
============================================================
IBM ML Professional Certificate — Supervised ML Extension

Extends the 3-class pipeline (Dry / Normal / Wet) to the full
WMO / McKee et al. (1993) SPEI severity scheme:

  Class            SPEI-3 range      Approx. %
  ─────────────────────────────────────────────
  Extremely Dry    < -2.0              0.97 %
  Severely Dry     -2.0 to -1.5        5.12 %
  Moderately Dry   -1.5 to -1.0       10.37 %
  Normal           -1.0 to +1.0       65.82 %
  Moderately Wet   +1.0 to +1.5       10.34 %
  Severely Wet     +1.5 to +2.0        5.69 %
  Extremely Wet    > +2.0              1.69 %

Key methodological differences vs. the 3-class pipeline
--------------------------------------------------------
1.  Target           : 7-class SPEI-3 scheme (vs. 3 classes)
2.  Primary metric   : Macro F1 (vs. weighted F1)
                       Macro treats every class equally regardless
                       of size — the honest metric when tail classes
                       matter (they do for insurance / agri risk)
3.  Class imbalance  : sample_weight="balanced" passed at fit-time
                       for Gradient Boosting (GBC has no class_weight
                       parameter — this is the correct fix)
4.  Cross-validation : macro F1 throughout (not weighted)
5.  ROC-AUC          : macro average (not weighted)

Same feature set, same 15 features, same train/test split seed.
Results are directly comparable to the 3-class pipeline.

Run from the folder that contains master_dataset.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics            import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay,
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import joblib

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
FIG_DPI = 150

# 7-class colour palette: red spectrum → grey → blue spectrum
CLASS_COLORS = {
    "Extremely Dry"  : "#8B0000",
    "Severely Dry"   : "#D4693A",
    "Moderately Dry" : "#F4A460",
    "Normal"         : "#6BAB6E",
    "Moderately Wet" : "#5BA3C9",
    "Severely Wet"   : "#2166AC",
    "Extremely Wet"  : "#08306B",
}

# Fixed class order — critical for LabelEncoder consistency
CLASS_NAMES = [
    "Extremely Dry", "Severely Dry", "Moderately Dry",
    "Normal",
    "Moderately Wet", "Severely Wet", "Extremely Wet",
]

SHORT_NAMES = ["ExDry", "SevDry", "ModDry", "Normal",
               "ModWet", "SevWet", "ExWet"]


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
# 2. DERIVE 7-CLASS TARGET
# =============================================================================

print("\n" + "=" * 70)
print("2. DERIVE 7-CLASS TARGET FROM SPEI-3")
print("=" * 70)

def classify_spei_7(spei: float) -> str:
    """Map a SPEI-3 value to one of seven WMO drought severity classes.

    Boundaries follow McKee et al. (1993) and are symmetric around zero.

    Args:
        spei: SPEI-3 value (standardised, dimensionless).

    Returns:
        String class label.
    """
    if   spei < -2.0:  return "Extremely Dry"
    elif spei < -1.5:  return "Severely Dry"
    elif spei < -1.0:  return "Moderately Dry"
    elif spei <=  1.0: return "Normal"
    elif spei <=  1.5: return "Moderately Wet"
    elif spei <=  2.0: return "Severely Wet"
    else:              return "Extremely Wet"

df["drought_class_7"] = df["spei_3"].apply(classify_spei_7)

print("\n7-class distribution:")
for cls in CLASS_NAMES:
    n = (df["drought_class_7"] == cls).sum()
    bar = "█" * int(n / 50)
    print(f"  {cls:<20} {n:5d}  ({100*n/len(df):5.2f}%)  {bar}")

print(f"\n  ⚠  Extremely Dry + Extremely Wet together: "
      f"{(df['drought_class_7'].isin(['Extremely Dry','Extremely Wet'])).sum()} rows "
      f"({100*(df['drought_class_7'].isin(['Extremely Dry','Extremely Wet'])).mean():.2f}%)")
print("     These rare classes will show lower F1 — this is expected and")
print("     reflects the physical reality that extreme events are rare.")


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ── 3a. Class distribution bar chart + SPEI histogram ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
counts = [( df["drought_class_7"] == c).sum() for c in CLASS_NAMES]
colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
bars = axes[0].bar(SHORT_NAMES, counts, color=colors,
                   edgecolor="white", linewidth=0.8)
axes[0].set_title("7-class drought distribution", fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].set_xlabel("Drought class")
for bar, n in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"{100*n/len(df):.1f}%",
                 ha="center", fontsize=8.5)

# SPEI-3 histogram coloured by class
for cls in CLASS_NAMES:
    vals = df[df["drought_class_7"] == cls]["spei_3"].dropna()
    axes[1].hist(vals, bins=40, alpha=0.65,
                 color=CLASS_COLORS[cls], label=cls, edgecolor="none")

# Draw SPEI boundary lines
for threshold, label in [(-2.0, ""), (-1.5, ""), (-1.0, "±1.0"),
                          (1.0, ""), (1.5, ""), (2.0, "")]:
    axes[1].axvline(threshold, color="gray", linestyle="--",
                    linewidth=0.8, alpha=0.7)
axes[1].set_title("SPEI-3 distribution by 7-class label", fontweight="bold")
axes[1].set_xlabel("SPEI-3")
axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=7.5, ncol=2)

plt.tight_layout()
plt.savefig("7cls_eda_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: 7cls_eda_distribution.png")

# ── 3b. Class mix by calendar month ───────────────────────────────────────────
month_class = (df.groupby(["month", "drought_class_7"])
               .size().unstack(fill_value=0)
               .reindex(columns=CLASS_NAMES))
month_class_pct = month_class.div(month_class.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(12)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
for cls in CLASS_NAMES:
    vals = month_class_pct[cls].values
    ax.bar(range(1, 13), vals, bottom=bottom,
           color=CLASS_COLORS[cls], label=cls,
           edgecolor="white", linewidth=0.4)
    bottom += vals

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_ylabel("% of month total")
ax.set_xlabel("Calendar month")
ax.set_title("7-class drought composition by calendar month", fontweight="bold")
ax.legend(loc="upper right", fontsize=8.5, ncol=2,
          bbox_to_anchor=(1.18, 1.0))
plt.tight_layout()
plt.savefig("7cls_eda_monthly.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: 7cls_eda_monthly.png")

# ── 3c. Comparison: 3-class vs 7-class split ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 3-class (from original)
three_counts = df["drought_class"].value_counts().reindex(["Dry","Normal","Wet"])
three_colors = ["#D4693A","#6BAB6E","#3A78B5"]
axes[0].bar(["Dry","Normal","Wet"], three_counts.values,
            color=three_colors, edgecolor="white")
axes[0].set_title("3-class scheme", fontweight="bold")
axes[0].set_ylabel("Count")
for i, (cls, v) in enumerate(three_counts.items()):
    axes[0].text(i, v + 50, f"{100*v/len(df):.1f}%", ha="center", fontsize=10)

# 7-class
axes[1].bar(SHORT_NAMES, counts, color=colors, edgecolor="white")
axes[1].set_title("7-class scheme (this pipeline)", fontweight="bold")
axes[1].set_ylabel("Count")
for i, (n, c) in enumerate(zip(counts, CLASS_NAMES)):
    axes[1].text(i, n + 30, f"{100*n/len(df):.1f}%",
                 ha="center", fontsize=7.5, rotation=45)

plt.suptitle("Class scheme comparison — same SPEI-3 data", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("7cls_eda_scheme_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: 7cls_eda_scheme_comparison.png")


# =============================================================================
# 4. PRE-PROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("4. PRE-PROCESSING")
print("=" * 70)

# ── 4a. Feature selection (identical to 3-class pipeline) ─────────────────────
EXCLUDE = [
    "station", "year", "month",
    "drought_class",       # 3-class target — exclude from features
    "drought_class_7",     # 7-class target — this is y
    "spei_3",              # direct source of target → must exclude
    "spei_6", "spei_12", "spei_24",
    "spei_1",
]
FEATURES = [c for c in df.columns if c not in EXCLUDE]
print(f"\nFeatures used ({len(FEATURES)}): {FEATURES}")
print("  (Identical feature set to 3-class pipeline — results are comparable)")

X = df[FEATURES].copy()
y_raw = df["drought_class_7"].copy()

# ── 4b. Encode target with fixed class order ───────────────────────────────────
le = LabelEncoder()
le.classes_ = np.array(CLASS_NAMES)
y_enc = le.transform(y_raw)
print(f"\nLabel encoding:")
for cls, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {code}  →  {cls}")

# ── 4c. Train / test split (stratified 80/20, same seed as 3-class) ───────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.20,
    random_state=42,
    stratify=y_enc,
)
print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"\nPer-class split:")
for i, cls in enumerate(CLASS_NAMES):
    n_tr = (y_train == i).sum()
    n_te = (y_test  == i).sum()
    print(f"  {cls:<20}  train={n_tr:4d}  test={n_te:3d}")

# ── 4d. Sample weights for Gradient Boosting ──────────────────────────────────
# GradientBoostingClassifier has NO class_weight parameter.
# The correct approach is compute_sample_weight, passed at fit-time.
# This was a bug in the 3-class pipeline — fixed here.
sample_weights_train = compute_sample_weight("balanced", y_train)
print(f"\nSample weights computed (balanced) — used for Gradient Boosting fit")
print(f"  Weight range: {sample_weights_train.min():.3f} – {sample_weights_train.max():.3f}")

# Class weights for LR and RF (these support class_weight param directly)
cw = compute_class_weight("balanced",
                           classes=np.unique(y_train),
                           y=y_train)
class_weight_dict = {i: w for i, w in enumerate(cw)}
print(f"\nClass weights (balanced) for LR and RF:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<20}  weight = {cw[i]:.3f}")

# ── 4e. Pre-processing steps (shared by all pipelines) ────────────────────────
preprocessor_steps = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
]


# =============================================================================
# 5. BUILD MODEL PIPELINES
# =============================================================================

print("\n" + "=" * 70)
print("5. BUILD MODEL PIPELINES")
print("=" * 70)

pipelines = {
    "Logistic Regression": Pipeline(preprocessor_steps + [
        ("clf", LogisticRegression(
            max_iter=2000,           # more iterations for 7-class convergence
            class_weight=class_weight_dict,
            random_state=42,
            solver="lbfgs",   # multinomial handled automatically in sklearn >= 1.5
        ))
    ]),
    "Random Forest": Pipeline(preprocessor_steps + [
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,      # relaxed vs 3-class (rare classes need smaller leaves)
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,
        ))
    ]),
    "Gradient Boosting": Pipeline(preprocessor_steps + [
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
            # NOTE: no class_weight param — balanced via sample_weight at fit()
        ))
    ]),
}

print("\n  Logistic Regression : class_weight=balanced (direct param)")
print("  Random Forest       : class_weight=balanced (direct param)")
print("  Gradient Boosting   : sample_weight=balanced (passed at fit — correct fix)")


# =============================================================================
# 6. TRAIN & CROSS-VALIDATE
# =============================================================================

print("\n" + "=" * 70)
print("6. TRAIN MODELS  (5-fold stratified CV — macro F1)")
print("=" * 70)
print("\n  Primary metric: macro F1 (each class weighted equally)")
print("  This is stricter than weighted F1 — rare classes penalise the score")
print("  if the model ignores them.\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipe in pipelines.items():
    print(f"  Training {name}...")

    if name == "Gradient Boosting":
        # CV with sample weights requires manual fold loop
        fold_scores = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            Xf_tr, Xf_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            yf_tr, yf_val = y_train[tr_idx], y_train[val_idx]
            sw_tr = sample_weights_train[tr_idx]
            pipe_clone = Pipeline(preprocessor_steps + [
                ("clf", GradientBoostingClassifier(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=4, subsample=0.8, random_state=42,
                ))
            ])
            pipe_clone.fit(Xf_tr, yf_tr,
                           clf__sample_weight=sw_tr)
            preds = pipe_clone.predict(Xf_val)
            from sklearn.metrics import f1_score
            fold_scores.append(f1_score(yf_val, preds,
                                        average="macro", zero_division=0))
        scores = np.array(fold_scores)
        # Fit final model on full training set with sample weights
        pipe.fit(X_train, y_train,
                 clf__sample_weight=sample_weights_train)
    else:
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="f1_macro", n_jobs=-1)
        pipe.fit(X_train, y_train)

    cv_results[name] = scores
    print(f"    CV Macro F1: {scores.mean():.4f} ± {scores.std():.4f}  ✓")


# =============================================================================
# 7. EVALUATE & COMPARE
# =============================================================================

print("\n" + "=" * 70)
print("7. EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)

# Binarise test labels once for ROC-AUC
y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))

results = {}
for name, pipe in pipelines.items():
    y_pred  = pipe.predict(X_test)
    y_prob  = pipe.predict_proba(X_test)

    acc      = accuracy_score(y_test, y_pred)
    f1_macro = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    y_test, y_pred, average="weighted", zero_division=0)
    roc_auc  = roc_auc_score(y_test_bin, y_prob,
                              multi_class="ovr", average="macro")
    report_dict = classification_report(y_test, y_pred,
                                        target_names=CLASS_NAMES,
                                        output_dict=True,
                                        zero_division=0)
    results[name] = {
        "accuracy"    : acc,
        "f1_macro"    : f1_macro,
        "f1_weighted" : f1_weighted,
        "roc_auc"     : roc_auc,
        "report"      : report_dict,
        "y_pred"      : y_pred,
        "y_prob"      : y_prob,
        "cv_f1_mean"  : cv_results[name].mean(),
        "cv_f1_std"   : cv_results[name].std(),
    }

    print(f"\n── {name} ──")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Macro F1  ★      : {f1_macro:.4f}  ← primary metric")
    print(f"  Weighted F1      : {f1_weighted:.4f}")
    print(f"  ROC-AUC (macro)  : {roc_auc:.4f}")
    print(f"  CV Macro F1      : {cv_results[name].mean():.4f} ± {cv_results[name].std():.4f}")
    print(f"\n  Per-class classification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=CLASS_NAMES,
                                 zero_division=0))


# ── 7a. Model comparison table ────────────────────────────────────────────────
print("\n── Model comparison summary ──")
metrics_df = pd.DataFrame({
    name: {
        "CV Macro F1 (mean)" : f"{r['cv_f1_mean']:.4f}",
        "CV Macro F1 (std)"  : f"± {r['cv_f1_std']:.4f}",
        "Test Accuracy"      : f"{r['accuracy']:.4f}",
        "Test Macro F1 ★"    : f"{r['f1_macro']:.4f}",
        "Test Weighted F1"   : f"{r['f1_weighted']:.4f}",
        "ROC-AUC (macro)"    : f"{r['roc_auc']:.4f}",
    }
    for name, r in results.items()
}).T
print(metrics_df.to_string())


# ── 7b. Confusion matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    # Normalise by true label (row) to show recall per class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SHORT_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(name, fontsize=10, fontweight="bold")
    # Annotate cells
    thresh = 0.5
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Confusion matrices (row-normalised) — 7-class test set",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("7cls_eval_confusion_matrices.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("\n  Saved: 7cls_eval_confusion_matrices.png")


# ── 7c. Metric comparison bar chart ──────────────────────────────────────────
metric_keys   = ["accuracy", "f1_macro", "f1_weighted", "roc_auc"]
metric_labels = ["Accuracy", "Macro F1 ★", "Weighted F1", "ROC-AUC"]
model_names   = list(results.keys())
x = np.arange(len(metric_keys))
width = 0.25
palette = ["#4878d0", "#ee854a", "#6acc65"]

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, color) in enumerate(zip(model_names, palette)):
    vals = [results[name][m] for m in metric_keys]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=color, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("7-class model performance comparison — test set", fontweight="bold")
ax.legend(loc="lower right")
ax.axhline(0.887, color="gray", linestyle=":", linewidth=1,
           label="3-class GB F1 (reference)")
ax.text(3.82, 0.892, "3-cls ref", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig("7cls_eval_model_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: 7cls_eval_model_comparison.png")


# ── 7d. Per-class F1 heatmap (all 3 models × 7 classes) ──────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
f1_matrix = np.array([
    [results[name]["report"].get(cls, {}).get("f1-score", 0)
     for cls in CLASS_NAMES]
    for name in model_names
])
im = ax.imshow(f1_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(model_names)))
ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(model_names, fontsize=9)
for i in range(len(model_names)):
    for j in range(len(CLASS_NAMES)):
        v = f1_matrix[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8.5, color="black" if 0.35 < v < 0.85 else "white")
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label="F1-score")
ax.set_title("Per-class F1-score by model — 7-class test set", fontweight="bold")
plt.tight_layout()
plt.savefig("7cls_eval_perclass_f1.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: 7cls_eval_perclass_f1.png")


# ── 7e. ROC curves ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, r) in zip(axes, results.items()):
    for i, cls in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i], r["y_prob"][:, i],
            name=cls, ax=ax,
            color=CLASS_COLORS[cls],
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"{name}\nAUC={r['roc_auc']:.3f}", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower right")
plt.suptitle("ROC curves (one-vs-rest, 7 classes) — test set",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("7cls_eval_roc_curves.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: 7cls_eval_roc_curves.png")


# =============================================================================
# 8. MODEL SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("8. MODEL SELECTION")
print("=" * 70)

best_name = max(results, key=lambda n: results[n]["f1_macro"])
best = results[best_name]
print(f"""
  Best model       : {best_name}
  Test Macro F1 ★  : {best['f1_macro']:.4f}
  Test Weighted F1 : {best['f1_weighted']:.4f}
  ROC-AUC (macro)  : {best['roc_auc']:.4f}
  Accuracy         : {best['accuracy']:.4f}
  CV Macro F1      : {best['cv_f1_mean']:.4f} ± {best['cv_f1_std']:.4f}

  Selection rationale
  ───────────────────
  Macro F1 is the primary criterion because all seven severity classes
  carry real-world consequence for drought risk management, even the
  rare extreme categories (<2% each). A model that ignores "Extremely
  Dry" conditions to boost overall accuracy is operationally dangerous
  for agricultural or insurance applications.

  Weighted F1 is reported alongside macro F1 as a secondary metric for
  comparison with the 3-class pipeline results.

  Expected performance gap vs. 3-class pipeline
  ───────────────────────────────────────────────
  The 3-class Gradient Boosting achieved F1=0.887 (weighted).
  The 7-class pipeline will show lower scores — this is expected and
  physically honest. "Extremely Dry" (111 rows, 0.97%) and "Extremely
  Wet" (194 rows, 1.69%) are inherently hard to classify because:
    (a) so few training examples exist for each
    (b) their neighbours in SPEI space ("Severely Dry/Wet") are similar
    (c) the model must distinguish SPEI < -2.0 from -2.0 to -1.5
        using only indirect climate features (not SPEI itself)
  Low F1 on extreme classes is a finding, not a failure.
""")


# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================

print("=" * 70)
print("9. FEATURE IMPORTANCE")
print("=" * 70)

best_pipe = pipelines[best_name]
clf = best_pipe.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

    print(f"\n  Top 15 features ({best_name}):")
    for feat, imp in feat_imp.head(15).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<22} {imp:.4f}  {bar}")

    # Compare with 3-class pipeline top feature
    print(f"\n  Note: precip_roll3 was dominant in 3-class pipeline (54.5%).")
    print(f"  In 7-class: {feat_imp['precip_roll3']:.4f} — "
          + ("still dominant" if feat_imp.index[0] == "precip_roll3"
             else f"rank {list(feat_imp.index).index('precip_roll3')+1}"))

    # Side-by-side: 3-class vs 7-class importance
    # (approximate 3-class values from the report for reference)
    ref_3cls = {
        "precip_roll3": 0.545, "t_med_lag1": 0.141, "t_max": 0.064,
        "t_med": 0.037, "t_max_anom": 0.033, "precip_anom": 0.030,
        "t_min": 0.023, "t_med_anom": 0.020, "t_med_lag3": 0.019,
        "month_sin": 0.019,
    }
    common = [f for f in feat_imp.index if f in ref_3cls]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    top15 = feat_imp.head(15)
    colors_bar = sns.color_palette("viridis", len(top15))
    axes[0].barh(top15.index[::-1], top15.values[::-1],
                 color=colors_bar[::-1], edgecolor="white")
    axes[0].set_xlabel("Importance (mean decrease in impurity)")
    axes[0].set_title(f"7-class: {best_name}", fontweight="bold")

    # 3-class reference bar
    ref_series = pd.Series(ref_3cls).sort_values(ascending=False)
    colors_ref = sns.color_palette("crest", len(ref_series))
    axes[1].barh(ref_series.index[::-1], ref_series.values[::-1],
                 color=colors_ref[::-1], edgecolor="white")
    axes[1].set_xlabel("Importance (mean decrease in impurity)")
    axes[1].set_title("3-class reference (from report)", fontweight="bold")

    plt.suptitle("Feature importance: 7-class vs 3-class pipeline",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("7cls_feat_importance.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: 7cls_feat_importance.png")

elif hasattr(clf, "coef_"):
    # Logistic Regression
    coef = pd.DataFrame(clf.coef_, index=CLASS_NAMES, columns=FEATURES)
    print("\n  Logistic Regression coefficients (top 5 per class):")
    for cls in CLASS_NAMES:
        top5 = coef.loc[cls].abs().nlargest(5).index.tolist()
        print(f"  {cls:<20}: {top5}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for ax, cls in zip(axes, CLASS_NAMES):
        c = coef.loc[cls].sort_values()
        bar_colors = ["#D4693A" if v < 0 else "#3A78B5" for v in c]
        ax.barh(c.index, c.values, color=bar_colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_visible(False)
    plt.suptitle(f"Logistic Regression coefficients — 7-class",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("7cls_feat_importance.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: 7cls_feat_importance.png")


# =============================================================================
# 10. EXTREME CLASS DEEP-DIVE
# =============================================================================

print("\n" + "=" * 70)
print("10. EXTREME CLASS DEEP-DIVE")
print("=" * 70)
print("\n  Analysing where the model struggles most — extreme classes\n")

best_r = results[best_name]
best_pred = best_r["y_pred"]

# What does the model predict instead of Extremely Dry?
print(f"  'Extremely Dry' (true) → model predictions:")
ex_dry_mask = y_test == 0   # class index 0
predicted_as = pd.Series(best_pred[ex_dry_mask]).map(
    dict(enumerate(CLASS_NAMES))).value_counts()
for cls, n in predicted_as.items():
    print(f"    predicted as {cls:<20}: {n}")

print(f"\n  'Extremely Wet' (true) → model predictions:")
ex_wet_mask = y_test == 6
predicted_as_wet = pd.Series(best_pred[ex_wet_mask]).map(
    dict(enumerate(CLASS_NAMES))).value_counts()
for cls, n in predicted_as_wet.items():
    print(f"    predicted as {cls:<20}: {n}")

# SPEI-3 distribution within confusion zones
print("\n  Key insight: extreme classes are confused with their SPEI neighbours")
print("  (Extremely Dry ↔ Severely Dry; Extremely Wet ↔ Severely Wet)")
print("  This is physically expected — the SPEI boundary at ±2.0 is")
print("  a statistical threshold, not a physical discontinuity.")


# =============================================================================
# 11. SAVE ARTEFACTS
# =============================================================================

print("\n" + "=" * 70)
print("11. SAVE ARTEFACTS")
print("=" * 70)

# Save best model
model_fname = f"7cls_best_model_{best_name.replace(' ', '_')}.pkl"
joblib.dump(best_pipe, model_fname)
print(f"  Saved model       : {model_fname}")

# Save label encoder
joblib.dump(le, "7cls_label_encoder.pkl")
print(f"  Saved encoder     : 7cls_label_encoder.pkl")

# Save metrics table
metrics_df.to_csv("7cls_model_comparison_metrics.csv")
print(f"  Saved metrics     : 7cls_model_comparison_metrics.csv")

# Save per-class F1 table for all models
perclass_rows = []
for name, r in results.items():
    for cls in CLASS_NAMES:
        row = r["report"].get(cls, {})
        perclass_rows.append({
            "model"    : name,
            "class"    : cls,
            "precision": round(row.get("precision", 0), 4),
            "recall"   : round(row.get("recall", 0), 4),
            "f1-score" : round(row.get("f1-score", 0), 4),
            "support"  : int(row.get("support", 0)),
        })
perclass_df = pd.DataFrame(perclass_rows)
perclass_df.to_csv("7cls_perclass_metrics.csv", index=False)
print(f"  Saved per-class   : 7cls_perclass_metrics.csv")

# Save dataset with 7-class label appended
df.to_csv("master_dataset_7class.csv", index=False)
print(f"  Saved dataset     : master_dataset_7class.csv")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("7-CLASS PIPELINE COMPLETE")
print("=" * 70)
print(f"""
Dataset    : 11,452 rows × {len(FEATURES)} features · 7 classes
Target     : drought_class_7 derived from SPEI-3 (WMO scheme)
Best model : {best_name}
  Macro F1 : {best['f1_macro']:.4f}   ← primary (class-equal)
  Wtd F1   : {best['f1_weighted']:.4f}   ← secondary (for comparison)
  AUC      : {best['roc_auc']:.4f}
  Accuracy : {best['accuracy']:.4f}

3-class reference (Gradient Boosting):
  Wtd F1   : 0.887  |  AUC: 0.960

Expected interpretation
  The macro F1 drop relative to the 3-class pipeline reflects the
  genuine difficulty of separating rare extreme severity classes
  using only indirect climate features. This is not a model failure —
  it is an honest representation of classification difficulty at the
  SPEI distribution tails.

Output files
  7cls_eda_distribution.png         – 7-class count + SPEI histogram
  7cls_eda_monthly.png              – stacked class by month
  7cls_eda_scheme_comparison.png    – 3-class vs 7-class side by side
  7cls_eval_confusion_matrices.png  – row-normalised confusion matrices
  7cls_eval_model_comparison.png    – metric bar chart (w/ 3-cls ref)
  7cls_eval_perclass_f1.png         – per-class F1 heatmap
  7cls_eval_roc_curves.png          – ROC curves (7 classes × 3 models)
  7cls_feat_importance.png          – importance vs 3-class comparison
  7cls_best_model_*.pkl             – serialised best pipeline
  7cls_label_encoder.pkl            – fitted LabelEncoder
  7cls_model_comparison_metrics.csv – summary table
  7cls_perclass_metrics.csv         – per-class precision/recall/F1
  master_dataset_7class.csv         – dataset with 7-class label added
 """)
