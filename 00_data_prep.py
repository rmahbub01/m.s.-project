"""
00_data_prep.py
----------------
Project : Determinants of Financial
          Worries in SAARC Countries
Purpose : Load the FULL Global Findex-style SAARC microdata extract
          (200 raw survey columns), recode the multinomial dependent
          variable (fin45), standardise missing-value tokens, and select
          an expanded, leakage-free predictor set for the analytic file
          used by every downstream script.
"""

from imports import *

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)
RAW_PATH = Path("data/saarc_data_full.csv")
OUT_PATH = Path("data/saarc_clean.csv")
DV = "fin45_cat"

df = pd.read_csv(RAW_PATH)


target_map = {
    "For monthly expenses, such as food, housing, or bills": "Daily/Monthly Expenses",
    "For medical costs in case of a serious illness or accident": "Medical Emergency",
    "For school or education fees": "Education",
    "For their old age": "Old Age",
    "For their business": "Business",
    "Some other reason": "Other/Unspecified",
    "Don't know": "Other/Unspecified",
    "Refused": "Other/Unspecified",
}
df["fin45_cat"] = df["fin45"].map(target_map)
assert df["fin45_cat"].isna().sum() == 0, "Unmapped category found in fin45"

# Standardise missing tokens on the original 15-variable core set
df["fin24a"] = df["fin24a"].fillna("Not Applicable")
df["domestic_remittances"] = df["domestic_remittances"].fillna("Not Applicable")

dkref_tokens = ["DK/ref", "Don't know", "Refused"]
for col in ["pay_utilities", "domestic_remittances", "fin24a", "fin24"]:
    df[col] = df[col].replace(dkref_tokens, "Don't know/Refused")

df["pay_utilities"] = df["pay_utilities"].replace(
    {"Don't know/Refused": "Other/DK", "In some other way": "Other/DK"}
)
df["domestic_remittances"] = df["domestic_remittances"].replace(
    {"Don't know/Refused": "Not Applicable"}
)

df["economy"] = df["economy"].astype(str)
df["age"] = df["age"].astype(float)
df["age_c"] = (
    df["age"] - df["age"].mean()
) / 10.0  # scaled to decades for numerical stability
df["age_c_sq"] = df["age_c"] ** 2

# Recode the varialbes to 0/1
for col in [
    "receive_wages",
    "receive_transfers",
    "receive_pensions",
    "receive_agriculture",
]:
    df[col + "_recv"] = (~df[col].isin(["Did not receive", "DK/ref"])).astype(int)

# Additional digital-finance and risk-management indicators
df["dig_account_bin"] = (df["dig_account"] == "Yes").astype(int)
df["merchantpay_dig_bin"] = (df["merchantpay_dig"] == "Yes").astype(int)
df["has_insurance_bin"] = (df["fin42"] == "Yes").astype(int)
df["has_mobile_phone_bin"] = (df["con1"] == "Yes").astype(int)


#     "could not afford medical care (fin28) / school fees in the past year (fin29)".
#     These are retained as explicit 3-level categoricals (Not
#     Applicable / No / Yes) because the skip pattern (only asked of
#     respondents who actually needed the service) is itself informative.
df["fin28_cat"] = df["fin28"].fillna("Not Applicable")
df["fin29_cat"] = df["fin29"].fillna("Not Applicable")

# Save cleaned analytic file
df.to_csv(OUT_PATH, index=False)
print(df.shape)
print("fin45_cat")
print(df["fin45_cat"].value_counts())
print("\nCountry distribution:\n", df["economy"].value_counts())
