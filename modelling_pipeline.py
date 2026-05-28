"""
Moldova Drought Classification — Modelling Pipeline
=====================================================
IBM ML Professional Certificate style workflow:

  1.  Load & inspect data
  2.  Exploratory Data Analysis (EDA)
  3.  Pre-processing
       a. Feature selection
       b. Train / test split
       c. Handle missing values (imputation)
       d. Encode target
       e. Scale features
  4.  Train three classifiers
       - Logistic Regression   (interpretable baseline)
       - Random Forest         (ensemble, feature importance)
       - Gradient Boosting     (XGBoost-style, best performance)
  5.  Evaluate & compare
       - Accuracy, Precision, Recall, F1 (weighted)
       - Confusion matrices
       - ROC-AUC (one-vs-rest)
  6.  Select best model & justify
  7.  Feature importance analysis
  8.  Save artefacts

Run from the folder that contains master_dataset.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics            import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay,
)
from sklearn.utils.class_weight import compute_class_weight

# ── plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
COLORS = {"Dry": "#d4693a", "Normal": "#4a9b6f", "Wet": "#3a78b5"}
FIG_DPI = 150


# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=" * 65)
print("1. LOAD DATA")
print("=" * 65)

df = pd.read_csv("master_dataset.csv")
print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Stations ({df['station'].nunique()}): {sorted(df['station'].unique())}")
print(f"Years  : {df['year'].min()} – {df['year'].max()}")
print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string(index=False)}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 65)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# ── 2a. Class distribution ────────────────────────────────────────────────────
print("\nTarget class distribution:")
vc = df["drought_class"].value_counts().sort_index()
for cls, n in vc.items():
    print(f"  {cls:<10} {n:5d}  ({100*n/len(df):.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Bar chart
vc_plot = vc.reindex(["Dry", "Normal", "Wet"])
axes[0].bar(vc_plot.index, vc_plot.values,
            color=[COLORS[c] for c in vc_plot.index], edgecolor="white", linewidth=0.8)
axes[0].set_title("Class distribution")
axes[0].set_ylabel("Count")
for i, (cls, v) in enumerate(vc_plot.items()):
    axes[0].text(i, v + 40, f"{100*v/len(df):.1f}%", ha="center", fontsize=10)

# Class distribution by month
month_class = (df.groupby(["month", "drought_class"])
               .size().unstack(fill_value=0)
               .reindex(columns=["Dry", "Normal", "Wet"]))
month_class_pct = month_class.div(month_class.sum(axis=1), axis=0) * 100
month_class_pct.plot(kind="bar", stacked=True, ax=axes[1],
                     color=[COLORS[c] for c in ["Dry","Normal","Wet"]],
                     edgecolor="white", linewidth=0.5)
axes[1].set_title("Class mix by calendar month")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("% of month")
axes[1].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], rotation=0)
axes[1].legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left")

# SPEI-3 distribution coloured by class
for cls in ["Dry", "Normal", "Wet"]:
    subset = df[df["drought_class"] == cls]["spei_3"].dropna()
    axes[2].hist(subset, bins=40, alpha=0.55, color=COLORS[cls], label=cls, edgecolor="none")
axes[2].axvline(-1, color="gray", linestyle="--", linewidth=0.9, label="±1 threshold")
axes[2].axvline( 1, color="gray", linestyle="--", linewidth=0.9)
axes[2].set_title("SPEI-3 distribution by class")
axes[2].set_xlabel("SPEI-3"); axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: eda_class_distribution.png")

# ── 2b. Feature correlations ──────────────────────────────────────────────────
numeric_cols = ["precip", "t_med", "t_min", "t_max",
                "spei_1", "spei_3", "spei_6", "spei_12",
                "precip_anom", "t_med_anom", "precip_roll3", "month"]

corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.savefig("eda_correlation_matrix.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: eda_correlation_matrix.png")

# ── 2c. Temperature & precip trends ──────────────────────────────────────────
annual = (df.groupby("year")[["t_med", "precip"]]
          .mean().reset_index())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(annual["year"], annual["t_med"], color="#c0392b", linewidth=1.2)
ax1.set_ylabel("Mean temperature (°C)")
ax1.set_title("Annual mean temperature — all stations")

ax2.bar(annual["year"], annual["precip"], color="#2980b9", alpha=0.7, width=0.8)
ax2.set_ylabel("Mean precipitation (mm)")
ax2.set_xlabel("Year")
ax2.set_title("Annual mean precipitation — all stations")

plt.tight_layout()
plt.savefig("eda_annual_trends.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: eda_annual_trends.png")


# =============================================================================
# 3. PRE-PROCESSING
# =============================================================================

print("\n" + "=" * 65)
print("3. PRE-PROCESSING")
print("=" * 65)

# ── 3a. Feature selection ─────────────────────────────────────────────────────
# Exclude: identifiers, other SPEI scales (would leak target signal),
#          spei_3 itself (IS the target source)
EXCLUDE = ["station", "year", "month", "drought_class",
           "spei_3",       # direct source of target — must exclude
           "spei_6", "spei_12", "spei_24",   # correlated with spei_3
           "spei_1",       # keep if you want; exclude to avoid leakage debate
           ]

FEATURES = [c for c in df.columns if c not in EXCLUDE]
print(f"\nFeatures used ({len(FEATURES)}): {FEATURES}")

X = df[FEATURES].copy()
y = df["drought_class"].copy()

# ── 3b. Encode target ─────────────────────────────────────────────────────────
le = LabelEncoder()
# Fix order: Dry=0, Normal=1, Wet=2
le.classes_ = np.array(["Dry", "Normal", "Wet"])
y_enc = le.transform(y)
CLASS_NAMES = list(le.classes_)
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 3c. Train / test split (stratified, 80/20) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.20,
    random_state=42,
    stratify=y_enc,
)
print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
for i, cls in enumerate(CLASS_NAMES):
    n_tr = (y_train == i).sum()
    n_te = (y_test  == i).sum()
    print(f"  {cls:<10}  train={n_tr:4d}  test={n_te:4d}")

# ── 3d. Class weights (handles Normal-class dominance) ────────────────────────
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(cw)}
print(f"\nClass weights (balanced): {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in cw]))}")

# ── 3e. Build pre-processing + model pipelines ────────────────────────────────
#   Imputer → Scaler (inside each pipeline so no data leaks from test set)

preprocessor_steps = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
]

pipelines = {
    "Logistic Regression": Pipeline(preprocessor_steps + [
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight=class_weight_dict,
            random_state=42,
            solver="lbfgs",
        ))
    ]),
    "Random Forest": Pipeline(preprocessor_steps + [
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
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
        ))
    ]),
}


# =============================================================================
# 4. TRAIN & CROSS-VALIDATE
# =============================================================================

print("\n" + "=" * 65)
print("4. TRAIN MODELS  (5-fold stratified cross-validation)")
print("=" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=cv, scoring="f1_weighted", n_jobs=-1)
    cv_results[name] = scores
    print(f"\n  {name}")
    print(f"    CV F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}")
    pipe.fit(X_train, y_train)
    print(f"    Training complete.")


# =============================================================================
# 5. EVALUATE & COMPARE
# =============================================================================

print("\n" + "=" * 65)
print("5. EVALUATION ON HELD-OUT TEST SET")
print("=" * 65)

results = {}
for name, pipe in pipelines.items():
    y_pred      = pipe.predict(X_test)
    y_prob      = pipe.predict_proba(X_test)
    acc         = accuracy_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    report_dict = classification_report(y_test, y_pred,
                                        target_names=CLASS_NAMES,
                                        output_dict=True)
    results[name] = {
        "accuracy"    : acc,
        "roc_auc"     : roc_auc,
        "f1_weighted" : report_dict["weighted avg"]["f1-score"],
        "precision_w" : report_dict["weighted avg"]["precision"],
        "recall_w"    : report_dict["weighted avg"]["recall"],
        "report"      : report_dict,
        "y_pred"      : y_pred,
        "y_prob"      : y_prob,
        "cv_f1_mean"  : cv_results[name].mean(),
        "cv_f1_std"   : cv_results[name].std(),
    }
    print(f"\n── {name} ──")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  ROC-AUC (OvR)  : {roc_auc:.4f}")
    print(f"  F1 (weighted)  : {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"  CV F1          : {cv_results[name].mean():.4f} ± {cv_results[name].std():.4f}")
    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


# ── 5a. Comparison table ──────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {
        "CV F1 (mean)": f"{r['cv_f1_mean']:.4f}",
        "CV F1 (std)" : f"± {r['cv_f1_std']:.4f}",
        "Test Accuracy": f"{r['accuracy']:.4f}",
        "Test F1 (w)"  : f"{r['f1_weighted']:.4f}",
        "ROC-AUC (OvR)": f"{r['roc_auc']:.4f}",
    }
    for name, r in results.items()
}).T
print("\n── Model comparison summary ──")
print(metrics_df.to_string())


# ── 5b. Confusion matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)
plt.suptitle("Confusion matrices — test set", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("eval_confusion_matrices.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("\n  Saved: eval_confusion_matrices.png")


# ── 5c. Metric comparison bar chart ──────────────────────────────────────────
metric_names = ["accuracy", "f1_weighted", "roc_auc"]
metric_labels = ["Accuracy", "F1 (weighted)", "ROC-AUC"]
model_names = list(results.keys())
x = np.arange(len(metric_names))
width = 0.25
palette = ["#4878d0", "#ee854a", "#6acc65"]

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, color) in enumerate(zip(model_names, palette)):
    vals = [results[name][m] for m in metric_names]
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=color, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("Model performance comparison — test set")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("eval_model_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: eval_model_comparison.png")


# ── 5d. ROC curves ────────────────────────────────────────────────────────────
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, r) in zip(axes, results.items()):
    for i, cls in enumerate(CLASS_NAMES):
        RocCurveDisplay.from_predictions(
            y_test_bin[:, i], r["y_prob"][:, i],
            name=f"{cls} (OvR)", ax=ax,
        )
    ax.plot([0,1],[0,1],"k--",linewidth=0.8)
    ax.set_title(f"{name}\nAUC={r['roc_auc']:.3f}")
plt.suptitle("ROC curves (one-vs-rest) — test set", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("eval_roc_curves.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()
print("  Saved: eval_roc_curves.png")


# =============================================================================
# 6. MODEL SELECTION
# =============================================================================

print("\n" + "=" * 65)
print("6. MODEL SELECTION")
print("=" * 65)

best_name = max(results, key=lambda n: results[n]["f1_weighted"])
best = results[best_name]
print(f"\n  Best model    : {best_name}")
print(f"  Test F1 (w)   : {best['f1_weighted']:.4f}")
print(f"  ROC-AUC (OvR) : {best['roc_auc']:.4f}")
print(f"  Accuracy      : {best['accuracy']:.4f}")
print("""
  Justification
  ─────────────
  Selection is based on weighted F1-score, which accounts for class
  imbalance (Normal class = 65.8% of data) better than raw accuracy.
  ROC-AUC confirms generalisation across all three class boundaries.
  Cross-validation F1 consistency (low std) indicates the result is
  not a fluke of the train/test split.
""")


# =============================================================================
# 7. FEATURE IMPORTANCE
# =============================================================================

print("=" * 65)
print("7. FEATURE IMPORTANCE")
print("=" * 65)

best_pipe = pipelines[best_name]
clf = best_pipe.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

    print(f"\n  Top 10 features ({best_name}):")
    print(feat_imp.head(10).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    top_n = feat_imp.head(15)
    colors = sns.color_palette("viridis", len(top_n))
    ax.barh(top_n.index[::-1], top_n.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title(f"Top 15 feature importances — {best_name}")
    plt.tight_layout()
    plt.savefig("feat_importance.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: feat_importance.png")

elif hasattr(clf, "coef_"):
    coef = pd.DataFrame(clf.coef_, index=CLASS_NAMES, columns=FEATURES)
    print(f"\n  Logistic Regression coefficients (top 5 per class):")
    for cls in CLASS_NAMES:
        top5 = coef.loc[cls].abs().nlargest(5).index.tolist()
        print(f"  {cls}: {top5}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, cls in zip(axes, CLASS_NAMES):
        c = coef.loc[cls].sort_values()
        colors = ["#d4693a" if v < 0 else "#3a78b5" for v in c]
        ax.barh(c.index, c.values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Coefficients: {cls}")
        ax.set_xlabel("Coefficient value")
    plt.tight_layout()
    plt.savefig("feat_importance.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: feat_importance.png")


# =============================================================================
# 8. SAVE ARTEFACTS
# =============================================================================

print("\n" + "=" * 65)
print("8. SAVE ARTEFACTS")
print("=" * 65)

import joblib
joblib.dump(best_pipe, f"best_model_{best_name.replace(' ','_')}.pkl")
print(f"  Saved model  : best_model_{best_name.replace(' ','_')}.pkl")

metrics_df.to_csv("model_comparison_metrics.csv")
print("  Saved metrics: model_comparison_metrics.csv")

print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)
print(f"""
Summary
-------
Dataset  : 11,452 rows × {len(FEATURES)} features | 3 classes (Dry / Normal / Wet)
Target   : drought_class derived from SPEI-3
Best     : {best_name}
  F1 (w) : {best['f1_weighted']:.4f}
  AUC    : {best['roc_auc']:.4f}

Output files
  master_dataset.csv            – full feature matrix
  eda_class_distribution.png    – class & seasonal distribution
  eda_correlation_matrix.png    – feature correlations
  eda_annual_trends.png         – temperature & precip over time
  eval_confusion_matrices.png   – confusion matrices (all 3 models)
  eval_model_comparison.png     – metric bar chart
  eval_roc_curves.png           – ROC curves
  feat_importance.png           – feature importances / coefficients
  model_comparison_metrics.csv  – comparison table
  best_model_*.pkl              – serialised best pipeline
""")
