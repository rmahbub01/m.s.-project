"""
04_diagnostics.py
------------------
Diagnostic checks prior to multinomial logit estimation:
  - Pearson correlation matrix of the (numeric-encoded) design matrix
  - Variance Inflation Factors (VIF) for every predictor
A VIF > 10 (or, more conservatively, > 5) signals problematic
multicollinearity; results are reported in Chapter 4.
"""

from imports import *

sns.set_theme(style="white", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

FIG = Path("figures")
TAB = Path("tables")

X_train = pd.read_csv("data/X_train.csv").astype(float)

# Correlation heatmap
corr = X_train.corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.7},
    ax=ax,
)
ax.set_title("Figure 7. Correlation Matrix of Encoded Predictors", fontweight="bold")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig07_correlation_heatmap.png")
plt.close()

# Variance Inflation Factors
X_const = add_constant(X_train)
vif = pd.DataFrame(
    {
        "feature": X_const.columns,
        "VIF": [
            variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])
        ],
    }
)
vif = vif[vif["feature"] != "const"].sort_values("VIF", ascending=False)
vif.to_csv(TAB / "table_vif.csv", index=False)
print(vif.to_string(index=False))

max_vif = vif["VIF"].max()
print(
    f"\nMax VIF = {max_vif:.2f}  ->",
    "no serious multicollinearity (all VIF < 5)"
    if max_vif < 5
    else "moderate/serious multicollinearity detected",
)
