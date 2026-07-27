# Determinants of Financial Worries in SAARC Countries

> **Master of Science (M.S.) Dissertation**
> Department of Statistics, University of Chittagong

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Polars-Data%20Processing-blue)](https://pola.rs/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This repository contains the complete research workflow, source code, and analytical materials for my **Master of Science (M.S.) dissertation** in Statistics, University of Chittagong.

**Dissertation Title**

> **Determinants of Financial Worries in SAARC Countries: An Expanded Multinomial and Machine-Learning Approach with Country and Life-Cycle Effects**

The project investigates the socioeconomic, demographic, and financial determinants of financial worries among individuals across SAARC countries using nationally representative survey data. It combines classical statistical inference (multinomial logistic regression) with modern machine learning models, interpreted through Explainable AI (SHAP), to identify key predictors of financial concern while accounting for country and life-cycle effects.

---

## 🎯 Research Objectives

- Identify the major determinants of financial worries in SAARC countries.
- Examine demographic and socioeconomic patterns in financial concern.
- Compare financial worries across SAARC countries.
- Evaluate country and life-cycle effects on financial anxiety.
- Compare multinomial logistic regression against machine-learning models.
- Interpret model predictions using SHAP-based explainability.

---

## 📁 Repository Structure

The analysis is organized as a numbered sequence of standalone Python scripts, run in order, plus a consolidated notebook.

```text
m.s.-project/
│
├── data/                          # Raw and processed SAARC survey data
├── figures/                       # Generated plots and visualizations
├── tables/                        # Generated summary/output tables
│
├── imports.py                     # Shared imports and project-wide settings
├── 00_data_prep.py                # Data cleaning, recoding, and preprocessing
├── 01_eda.py                      # Univariate exploratory data analysis
├── 02_bivariate.py                # Bivariate / cross-tabulation analysis
├── 03_feature_engineering.py      # Feature construction and transformation
├── 04_diagnostics.py              # Model diagnostics and assumption checks
├── 05_multinomial_logit.py        # Multinomial logistic regression modeling
├── 06_ml_comparison.py            # Machine learning model comparison
├── 07_shap_analysis.py            # SHAP-based model interpretation
│
├── analysis.ipynb                 # Consolidated end-to-end notebook
├── metadata_SAARC_data_full.docx  # Variable/data dictionary for the SAARC dataset
└── README.md
```

---

## 📈 Analysis Pipeline

```text
data/ (raw survey data)
      │
      ▼
00_data_prep.py            → cleaning, recoding, preprocessing
      │
      ▼
01_eda.py                  → univariate exploratory analysis
      │
      ▼
02_bivariate.py            → bivariate / cross-tab analysis
      │
      ▼
03_feature_engineering.py  → feature construction
      │
      ▼
04_diagnostics.py          → model diagnostics
      │
      ▼
05_multinomial_logit.py    → multinomial logistic regression
      │
      ▼
06_ml_comparison.py        → Random Forest / XGBoost / LightGBM / CatBoost
      │
      ▼
07_shap_analysis.py        → SHAP interpretation
      │
      ▼
figures/ + tables/          (final outputs)
```

`analysis.ipynb` walks through this same pipeline end-to-end in a single notebook, useful for a linear read-through or presentation.

---

## 🛠 Technology Stack

**Language:** Python 3.13+

**Data processing:** Polars, NumPy, Pandas

**Statistical modeling:** statsmodels (multinomial logistic regression)

**Machine learning:** scikit-learn, XGBoost, LightGBM, CatBoost

**Model interpretation:** SHAP

**Visualization:** Plotnine, Matplotlib

**Tools:** Git, GitHub, Jupyter Notebook

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone git@github.com:rmahbub01/m.s.-project.git
cd m.s.-project
```

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` is not yet in the repo — see [Suggested Additions](#-suggested-additions) below.

### Run the pipeline

Scripts are numbered and meant to be run in order:

```bash
python 00_data_prep.py
python 01_eda.py
python 02_bivariate.py
python 03_feature_engineering.py
python 04_diagnostics.py
python 05_multinomial_logit.py
python 06_ml_comparison.py
python 07_shap_analysis.py
```

Or open `analysis.ipynb` to run the full workflow interactively in Jupyter.

---

## 📊 Data

The dataset consists of SAARC-country survey data on financial worries and related socioeconomic/demographic variables. Refer to **`metadata_SAARC_data_full.docx`** for the complete variable/data dictionary, including variable names, labels, and coding schemes.

Raw and processed files live under `data/`. If the original survey data are restricted or not redistributable, note that explicitly here (e.g., "raw survey microdata not included due to licensing; contact the author for access").

---

## 📌 Outputs

- **`figures/`** — exploratory and modeling visualizations (EDA plots, diagnostic plots, SHAP summary/dependence plots, etc.)
- **`tables/`** — regression output tables, odds ratios, model comparison metrics, and other summary tables used in the dissertation

---

## 📚 Citation

If you use this repository for research or educational purposes, please cite the associated dissertation:

```text
Rahman, M. M. (2026).
Determinants of Financial Worries in SAARC Countries:
An Expanded Multinomial and Machine-Learning Approach
with Country and Life-Cycle Effects.
Master of Science Dissertation,
Department of Statistics,
University of Chittagong.
```

---

## 👨‍💻 Author

**Md. Mahbub Rahman**
M.S. in Statistics, Department of Statistics, University of Chittagong, Bangladesh
GitHub: [@rmahbub01](https://github.com/rmahbub01)

---

## 📜 License

Released under the **MIT License** unless stated otherwise. Add a `LICENSE` file to the repo root to make this explicit.

---

## ✅ Suggested Additions

To make the repository more complete and citable:

- [ ] `requirements.txt` (or `pyproject.toml`) listing exact package versions used
- [ ] `LICENSE` file (MIT or your preferred license)
- [ ] `CITATION.cff` so GitHub auto-generates a "Cite this repository" button
- [ ] Short docstring/header comment at the top of each numbered script describing its inputs/outputs
- [ ] A `.gitignore` for `data/` raw files, `__pycache__/`, and notebook checkpoints if not already present
