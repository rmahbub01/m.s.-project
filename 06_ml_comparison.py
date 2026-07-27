"""
06_ml_comparison.py
---------------------
Trains and evaluates three machine-learning classifiers on the identical
train/test split used for the multinomial logit (script 05):
  1. Support Vector Machine (RBF kernel, probability=True)
  2. Random Forest
  3. XGBoost (multi:softprob)
Reports accuracy, macro-F1, weighted-F1, and multiclass ROC-AUC (OvR),
saves confusion matrices, and writes a consolidated model-comparison
table + bar chart that also incorporates the multinomial logit results
from script 05.
"""

from imports import *

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

FIG = Path("figures")
TAB = Path("tables")

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
class_names = le.classes_
joblib.dump(le, "data/label_encoder.pkl")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results = []
fitted_models = {}

# Support Vector Machine (RBF kernel)
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42,
)
svm.fit(X_train_sc, y_train_enc)
pred_svm = svm.predict(X_test_sc)
proba_svm = svm.predict_proba(X_test_sc)
fitted_models["SVM"] = svm

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train_enc)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)
fitted_models["Random Forest"] = rf

# XGBoost
sample_weight = (
    pd.Series(y_train_enc)
    .map((1 / pd.Series(y_train_enc).value_counts(normalize=True)))
    .values
)

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(class_names),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train_enc, sample_weight=sample_weight)
pred_xgb = xgb.predict(X_test)
proba_xgb = xgb.predict_proba(X_test)
fitted_models["XGBoost"] = xgb

joblib.dump(fitted_models, "data/fitted_ml_models.pkl")
joblib.dump(scaler, "data/scaler.pkl")


# Evaluation helper
def evaluate(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan
    print(f"\n=== {name} ===")
    print(
        f"Accuracy={acc:.4f}  Macro-F1={f1m:.4f}  Weighted-F1={f1w:.4f}  Macro-AUC={auc:.4f}"
    )
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    results.append(
        {
            "model": name,
            "accuracy": acc,
            "f1_macro": f1m,
            "f1_weighted": f1w,
            "roc_auc_macro": auc,
        }
    )
    return confusion_matrix(y_true, y_pred)


cm_svm = evaluate("Support Vector Machine (RBF)", y_test_enc, pred_svm, proba_svm)
cm_rf = evaluate("Random Forest", y_test_enc, pred_rf, proba_rf)
cm_xgb = evaluate("XGBoost", y_test_enc, pred_xgb, proba_xgb)

# # Confusion matrices figure
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
models = [
    ("SVM (RBF)", cm_svm),
    ("Random Forest", cm_rf),
    ("XGBoost", cm_xgb),
]

for ax, (title, cm) in zip(axes, models):
    # Row-normalized percentages
    cm_pct = cm.astype(float)
    cm_pct = cm_pct / cm_pct.sum(axis=1, keepdims=True) * 100

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.8,
        linecolor="white",
        square=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(
        "Predicted Label",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "True Label",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )
    ax.tick_params(axis="x", labelrotation=30, labelsize=10)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)

plt.suptitle(
    "Figure 8. Row-normalized Confusion Matrices (%) on Test Set",
    fontsize=18,
    fontweight="bold",
    y=1.02,
)
plt.savefig(
    FIG / "fig08_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
)

# Consolidated model-comparison table + chart (adds mnlogit from script 05)
ml_results = pd.DataFrame(results)
mnlogit_row = pd.read_csv(TAB / "modelcomp_mnlogit.csv")
comparison = pd.concat([mnlogit_row, ml_results], ignore_index=True)
comparison.to_csv(TAB / "table_model_comparison.csv", index=False)
print("\nModel comparison:\n", comparison)

# Random Forest feature importance (native) for cross-check against SHAP
rf_imp = pd.DataFrame(
    {"feature": X_train.columns, "importance": rf.feature_importances_}
)
rf_imp = rf_imp.sort_values("importance", ascending=False)
rf_imp.to_csv(TAB / "table_rf_feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 9))
top15 = rf_imp.head(15).sort_values("importance")
ax.barh(top15["feature"], top15["importance"], color="#55A868")
ax.set_title("Figure 9. Random Forest Feature Importance (Top 15)")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig(FIG / "fig09_rf_feature_importance.png")

print("\nML comparison complete.")
