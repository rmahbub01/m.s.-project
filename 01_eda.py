"""
01_eda.py
---------
Exploratory Data Analysis (EDA).
Produces: missingness table, frequency tables for the dependent variable
and every predictor, and publication-ready univariate bar charts.
"""

from imports import *

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    }
)

FIG = Path("figures")
TAB = Path("tables")
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

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
]
new_predictors = [
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

used_columns = ["fin45"] + predictors + new_predictors

# Missingness table (based on ORIGINAL raw values, before recoding)
raw = pd.read_csv("data/saarc_data_full.csv")
raw = raw[used_columns]
miss = raw.isna().sum().to_frame("n_missing")
miss["pct_missing"] = (miss["n_missing"] / len(raw) * 100).round(2)
miss = miss[miss["n_missing"] > 0].sort_values("n_missing", ascending=False)
miss.to_csv(TAB / "table_missingness.csv")
print("Missingness:\n", miss)

# Frequency table for the dependent variable
target_freq = df["fin45_cat"].value_counts().to_frame("n")
target_freq["pct"] = (target_freq["n"] / len(df) * 100).round(2)

# Figure 1: distribution of the dependent variable
fig, ax = plt.subplots(figsize=(9, 6))
order = target_freq.index
sns.barplot(
    x=target_freq["pct"], y=order, hue=order, palette="crest", legend=False, ax=ax
)
for i, v in enumerate(target_freq["pct"]):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=12)
ax.set_xlabel("Share of respondents (%)")
ax.set_ylabel("")
ax.set_title("Figure 1. Distribution of Greatest Financial Worry (fin45)")
ax.set_xlim(0, target_freq["pct"].max() + 8)
plt.tight_layout()
plt.savefig(FIG / "fig01_target_distribution.png")
plt.close()

# Country and age distributions
country_freq = df["economy"].value_counts().to_frame("n")
country_freq["pct"] = (country_freq["n"] / len(df) * 100).round(2)
country_freq.to_csv(TAB / "table_freq_economy.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
sns.barplot(
    x=country_freq["pct"],
    y=country_freq.index,
    hue=country_freq.index,
    palette="crest",
    legend=False,
    ax=axes[0],
)
axes[0].set_title("Country (economy) composition of the pooled sample")
axes[0].set_xlabel("%")
sns.histplot(df["age"], bins=30, color="#4C72B0", ax=axes[1])
axes[1].set_title("Age distribution")
axes[1].set_xlabel("Age (years)")
fig.suptitle("Figure 2. Country Composition and Age Distribution", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "fig02_country_age.png")
plt.close()

# Frequency tables + bar charts for every predictor
all_freqs = {}
for col in predictors:
    freq = df[col].value_counts().to_frame("n")
    freq["pct"] = (freq["n"] / len(df) * 100).round(2)
    all_freqs[col] = freq
    freq.to_csv(TAB / f"table_freq_{col}.csv")

# Composite Figure 3: grid of predictor distributions (demographic /
# access block) to keep the report concise while still publication ready
demo_cols = ["female", "educ", "inc_q", "emp_in", "urbanicity"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(demo_cols):
    f = all_freqs[col].sort_values("pct", ascending=False)
    sns.barplot(
        x=f["pct"], y=f.index, hue=f.index, palette="mako", legend=False, ax=axes[i]
    )
    axes[i].set_title(col)
    axes[i].set_xlabel("%")
    axes[i].set_ylabel("")
axes[-1].axis("off")
fig.suptitle(
    "Figure 3. Distribution of Demographic and Labour-Market Predictors",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(FIG / "fig03_demographic_distributions.png")
plt.close()

fin_cols = [
    "account_fin",
    "account_mob",
    "saved",
    "borrowed",
    "anydigpayment",
    "internet_use",
]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(fin_cols):
    f = all_freqs[col].sort_values("pct", ascending=False)
    sns.barplot(
        x=f["pct"], y=f.index, hue=f.index, palette="flare", legend=False, ax=axes[i]
    )
    axes[i].set_title(col)
    axes[i].set_xlabel("%")
    axes[i].set_ylabel("")
fig.suptitle(
    "Figure 4. Distribution of Financial Inclusion and Digital Finance Indicators",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(FIG / "fig04_finance_distributions.png")
plt.close()

print("\nEDA complete. Figures written to figures/, tables written to tables/.")
