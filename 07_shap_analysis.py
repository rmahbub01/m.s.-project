"""
07_shap_analysis.py
---------------------
SHAP (SHapley Additive exPlanations) analysis of the best tree-based
classifier (XGBoost) to interpret feature contributions to each
financial-worry category. Produces a global mean(|SHAP|) importance
ranking (bar) and a class-specific beeswarm summary plot for the modal
outcome, "Daily/Monthly Expenses", plus the top competing outcome,
"Medical Emergency".
"""

from imports import *

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

FIG = Path("figures")
TAB = Path("tables")

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
le = joblib.load("data/label_encoder.pkl")
models = joblib.load("data/fitted_ml_models.pkl")
xgb = models["XGBoost"]
class_names = le.classes_

# Use a manageable background/explain sample for speed & readability
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)

# Normalise shape across shap versions: want a list of (n_samples, n_features) per class
if isinstance(shap_values, list):
    sv_list = shap_values
else:
    sv_arr = np.array(shap_values)
    if sv_arr.ndim == 3 and sv_arr.shape[-1] == len(class_names):
        sv_list = [sv_arr[:, :, k] for k in range(len(class_names))]
    else:
        sv_list = [sv_arr[k] for k in range(sv_arr.shape[0])]

# Global feature importance: mean(|SHAP|) averaged across all classes
mean_abs_per_class = np.stack(
    [np.abs(sv).mean(axis=0) for sv in sv_list]
)  # (n_classes, n_features)
global_importance = mean_abs_per_class.mean(axis=0)
imp_df = pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": global_importance})
imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
imp_df.to_csv(TAB / "table_shap_global_importance.csv", index=False)
print(imp_df.head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 9))
top15 = imp_df.head(15).sort_values("mean_abs_shap")
ax.barh(top15["feature"], top15["mean_abs_shap"], color="#C44E52")
ax.set_title(
    "Figure 10. Global SHAP Feature Importance — XGBoost\n(mean |SHAP value|, averaged across all outcome classes)"
)
ax.set_xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(FIG / "fig10_shap_global_importance.png")
plt.close()

# Class-specific beeswarm summary plots
focus_classes = ["Daily/Monthly Expenses", "Medical Emergency", "Education", "Business"]

imgs = []
for cls in focus_classes:
    idx = list(class_names).index(cls)
    plt.figure(figsize=(10, 9))
    shap.summary_plot(sv_list[idx], X_test, show=False, plot_size=(10, 9))
    plt.title(f"SHAP Summary — Outcome: {cls}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    imgs.append(Image.open(buf))

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
for ax, img, cls in zip(axes.flatten(), imgs, focus_classes):
    ax.imshow(img)
    ax.axis("off")

plt.tight_layout()
combined_path = FIG / "fig_shap_summary_combined.png"
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved combined SHAP summary plot: {combined_path}")
print("\nSHAP analysis complete.")
