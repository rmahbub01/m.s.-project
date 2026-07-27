"""
05_multinomial_logit.py
-------------------------
Estimates a multinomial logistic regression (MNLogit) of the greatest
financial worry (fin45_cat) on the demographic, labour-market and
financial-inclusion predictor set. "Daily/Monthly Expenses" (the modal
category) is the reference outcome. Reports coefficients, relative risk
ratios (RRR = exp(beta)), McFadden's pseudo-R2, and an in-sample /
out-of-sample classification performance summary so the parametric model
can be benchmarked against the machine-learning classifiers in script 06.
"""

from imports import *

TAB = Path("tables")

X_train = pd.read_csv("data/X_train.csv").astype(float)
X_test = pd.read_csv("data/X_test.csv").astype(float)
y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

# Reference category = modal class
REFERENCE = "Daily/Monthly Expenses"
categories = [REFERENCE] + [c for c in y_train.unique() if c != REFERENCE]
cat_dtype = pd.CategoricalDtype(categories=categories, ordered=False)
y_train_cat = y_train.astype(cat_dtype).cat.codes
y_test_cat = y_test.astype(cat_dtype).cat.codes

X_train_c = sm.add_constant(X_train)
X_test_c = sm.add_constant(X_test, has_constant="add")

model = sm.MNLogit(y_train_cat, X_train_c)
result = model.fit(method="bfgs", maxiter=2000, gtol=1e-5, disp=True)
print(result.summary())

# McFadden's pseudo-R2
llf = result.llf
llnull = result.llnull
pseudo_r2 = 1 - llf / llnull
print(f"\nMcFadden's pseudo-R2 = {pseudo_r2:.4f}")

# Relative risk ratios (RRR) table, one block per non-reference outcome
params = result.params
pvalues = result.pvalues
rrr = np.exp(params)
outcome_labels = [c for c in categories if c != REFERENCE]

rrr_tables = {}
for j, label in enumerate(outcome_labels):
    tbl = pd.DataFrame(
        {
            "coef": params.iloc[:, j],
            "RRR": rrr.iloc[:, j],
            "p_value": pvalues.iloc[:, j],
        }
    )
    tbl["sig"] = tbl["p_value"].apply(
        lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    )
    rrr_tables[label] = tbl
    tbl.to_csv(
        TAB / f"table_mnlogit_RRR_{label.replace('/', '_').replace(' ', '_')}.csv"
    )

print(
    "\nRRR table (vs. reference = 'Daily/Monthly Expenses') written for outcomes:",
    outcome_labels,
)


# Predictive performance (train and test)
def predict_labels(Xc):
    probs = result.predict(Xc)
    codes = np.argmax(probs.values, axis=1)
    return pd.Categorical.from_codes(codes, categories=categories)


pred_train = predict_labels(X_train_c)
pred_test = predict_labels(X_test_c)

acc_train = accuracy_score(y_train, pred_train)
acc_test = accuracy_score(y_test, pred_test)
f1_test_macro = f1_score(y_test, pred_test, average="macro")
f1_test_weighted = f1_score(y_test, pred_test, average="weighted")

print(f"\nTrain accuracy = {acc_train:.4f}")
print(f"Test accuracy  = {acc_test:.4f}")
print(f"Test macro-F1  = {f1_test_macro:.4f}")
print(f"Test weighted-F1 = {f1_test_weighted:.4f}")

report = classification_report(y_test, pred_test, digits=3)
print("\n", report)

# Macro-average one-vs-rest ROC-AUC (uses fitted probabilities), for a
# like-for-like comparison against the SVM / RF / XGBoost models in 06
proba_test = result.predict(X_test_c).values
try:
    auc_test = roc_auc_score(y_test_cat, proba_test, multi_class="ovr", average="macro")
except Exception:
    auc_test = np.nan
print(f"Test macro-AUC (OvR) = {auc_test:.4f}")

# Save model comparison row for later aggregation
pd.DataFrame(
    [
        {
            "model": "Multinomial Logistic Regression",
            "accuracy": acc_test,
            "f1_macro": f1_test_macro,
            "f1_weighted": f1_test_weighted,
            "roc_auc_macro": auc_test,
        }
    ]
).to_csv(TAB / "modelcomp_mnlogit.csv", index=False)

pd.DataFrame(
    [
        {
            "pseudo_r2_mcfadden": pseudo_r2,
            "llf": llf,
            "llnull": llnull,
            "n_obs": int(result.nobs),
        }
    ]
).to_csv(TAB / "table_mnlogit_fit.csv", index=False)
