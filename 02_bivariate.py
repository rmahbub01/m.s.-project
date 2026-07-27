"""
02_bivariate.py
----------------
Bivariate analysis of the dependent variable (fin45_cat) against every
categorical predictor: contingency tables, Pearson chi-square tests of
independence, and Cramer's V effect size. Also produces stacked
percentage-bar visualisations for the four predictors with the largest
association with financial worry.
"""

from imports import *

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300})

FIG = Path("figures")
TAB = Path("tables")

df = pd.read_csv("data/saarc_clean.csv")

predictors = [
    "female",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
    "account_fin",
    "account_mob",
    "saved",
    "borrowed",
    "anydigpayment",
    "internet_use",
    "fin24",
    "fin24a",
    "pay_utilities",
    "domestic_remittances",
    "economy",
    "receive_wages",
    "receive_transfers",
    "receive_pensions",
    "receive_agriculture",
    "dig_account",
    "merchantpay_dig",
    "con1",
    "fin28",
    "fin29",
]


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013)."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))))


results = []
for col in predictors:
    ct = pd.crosstab(df[col], df["fin45_cat"])
    chi2, p, dof, _ = chi2_contingency(ct)
    v = cramers_v(ct.values)
    results.append(
        {
            "predictor": col,
            "chi2": round(chi2, 2),
            "dof": dof,
            "p_value": p,
            "cramers_v": round(v, 3),
        }
    )

assoc = pd.DataFrame(results).sort_values("cramers_v", ascending=False)
assoc["p_value_fmt"] = assoc["p_value"].apply(
    lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
)

# Age (continuous) vs. fin45_cat: Kruskal-Wallis H-test (chi-square/
#     Cramer's V requires categorical predictors, so age is tested
#     separately) plus group-wise mean age.

groups = [g["age"].values for _, g in df.groupby("fin45_cat")]
h_stat, p_kw = kruskal(*groups)
age_means = df.groupby("fin45_cat")["age"].agg(["mean", "std", "count"]).round(1)
age_means.to_csv(TAB / "table_age_by_worry.csv")

# Figure 5: Cramer's V ranking (effect-size lollipop chart)
fig, ax = plt.subplots(figsize=(10, 8))
order = assoc.sort_values("cramers_v")
ax.hlines(
    y=order["predictor"], xmin=0, xmax=order["cramers_v"], color="#4C72B0", linewidth=2
)
ax.plot(order["cramers_v"], order["predictor"], "o", color="#C44E52", markersize=9)
ax.set_xlabel("Cramer's V (association with fin45_cat)")
ax.set_title(
    "Figure 5. Bivariate Association of Predictors with\nGreatest Financial Worry (Cramer's V)"
)
plt.tight_layout()
plt.savefig(FIG / "fig05_cramers_v_ranking.png")
plt.close()

# Figure 6: stacked 100% bar chart, top 4 predictors by Cramer's V
top4 = assoc["predictor"].head(4).tolist()
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(top4):
    ct = pd.crosstab(df[col], df["fin45_cat"], normalize="index") * 100
    ct = ct[df["fin45_cat"].value_counts().index]  # consistent class order
    ct.plot(kind="bar", stacked=True, ax=axes[i], colormap="viridis", legend=False)
    axes[i].set_title(
        f"{col}  (Cramer's V = {assoc.set_index('predictor').loc[col, 'cramers_v']:.3f})"
    )
    axes[i].set_ylabel("% within category")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=30)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
fig.suptitle(
    "Figure 6. Composition of Greatest Financial Worry by Top-4 Associated Predictors",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(FIG / "fig06_stacked_top4_predictors.png", bbox_inches="tight")
plt.close()

print("\nBivariate analysis complete.")
