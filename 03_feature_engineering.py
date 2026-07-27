"""
03_feature_engineering.py
--------------------------
Encodes predictors for modelling:
  - Ordinal encoding for genuinely ordinal variables (educ, inc_q, fin24a)
  - One-hot (dummy) encoding for nominal variables, including country identifier (economy) and two financial-stress
    history variables (fin28_cat, fin29_cat)
  - Binary (0/1) recoding for Yes/No-type indicators, including four
    income-source variables and three digital-finance/insurance
    indicators
  - Continuous age (mean-centred) plus a quadratic term to capture a
    possible non-linear (hump/U-shaped) life-cycle relationship with
    financial worry
Builds a stratified 70/30 train-test split and persists the design
matrices so that every model in scripts 04-07 is trained/evaluated on an
identical split (fair comparison across the multinomial logit, SVM,
Random Forest and XGBoost).
"""

from imports import *

df = pd.read_csv("data/saarc_clean.csv")

# Ordinal encodings
educ_order = {
    "Primary education or less": 0,
    "Secondary education": 1,
    "Tertiary education or more": 2,
}
incq_order = {"Poorest": 0, "Poor": 1, "Middle": 2, "Rich": 3, "Richest": 4}
fin24a_order = {
    "Not Applicable": -1,
    "Not difficult at all": 0,
    "Somewhat difficult": 1,
    "Very difficult": 2,
    "Don't know/Refused": 1,
}  # DK imputed at the sample median category

df["educ_ord"] = df["educ"].map(educ_order)
df["inc_q_ord"] = df["inc_q"].map(incq_order)
df["fin24a_ord"] = df["fin24a"].map(fin24a_order)

# Binary recodes (Yes = 1)
binary_cols = [
    "account_fin",
    "account_mob",
    "saved",
    "borrowed",
    "anydigpayment",
    "internet_use",
]
for col in binary_cols:
    df[col + "_bin"] = (df[col] == "Yes").astype(int)

df["female_bin"] = (df["female"] == "Female").astype(int)
df["urban_bin"] = (df["urbanicity"] == "Urban").astype(int)
df["employed_bin"] = (df["emp_in"] == "In the workforce").astype(int)

# Nominal one-hot encodings (drop_first to avoid the dummy trap)
nominal_cols = ["fin24", "pay_utilities", "domestic_remittances"]
dummies = pd.get_dummies(df[nominal_cols], prefix=nominal_cols, drop_first=True)

feature_cols_ordinal_binary = [
    "female_bin",
    "educ_ord",
    "inc_q_ord",
    "employed_bin",
    "urban_bin",
    "account_fin_bin",
    "account_mob_bin",
    "saved_bin",
    "borrowed_bin",
    "anydigpayment_bin",
    "internet_use_bin",
    "fin24a_ord",
]

# Additional features enabled by the full 200-column extract
new_binary_cols = [
    "receive_wages_recv",
    "receive_transfers_recv",
    "receive_pensions_recv",
    "receive_agriculture_recv",
    "dig_account_bin",
    "merchantpay_dig_bin",
    "has_mobile_phone_bin",
]
# Note: has_insurance_bin (fin42) was found to be perfectly collinear
# (r = 1.00, identical values) with receive_agriculture_recv in this
# extract and is therefore dropped to avoid infinite VIF; only one of
# the pair is retained.
feature_cols_ordinal_binary += new_binary_cols
feature_cols_ordinal_binary += ["age_c", "age_c_sq"]

# fin28/fin29 simplified to a single binary flag (Yes vs. No/Not-Applicable)
df["fin28_bin"] = (df["fin28_cat"] == "Yes").astype(int)
df["fin29_bin"] = (df["fin29_cat"] == "Yes").astype(int)
feature_cols_ordinal_binary += ["fin28_bin", "fin29_bin"]

new_nominal_cols = ["economy"]
new_dummies = pd.get_dummies(
    df[new_nominal_cols], prefix=new_nominal_cols, drop_first=True
)
dummies = pd.concat([dummies, new_dummies], axis=1)

X = pd.concat([df[feature_cols_ordinal_binary], dummies], axis=1)
X.columns = [c.replace(" ", "_").replace(",", "").replace("/", "_") for c in X.columns]
y = df["fin45_cat"]

print("Feature matrix shape:", X.shape)
print("Features:", X.columns.tolist())

# Stratified train/test split (70/30), fixed random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

Path("data").mkdir(exist_ok=True)
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
joblib.dump(list(X.columns), "data/feature_names.pkl")

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
