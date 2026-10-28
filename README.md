# LoanLens — Loan Repayment Prediction

A local Streamlit web app built from the supplied **Loan Repayment Prediction Using Ensemble Learning Methods** notebook and CSV dataset. It includes an interactive dashboard, exploratory charts, comparison of baseline classifiers, and a form for illustrative predictions.

## What this project predicts

The dataset target is `not.fully.paid`:
- `0` = fully paid in the historical data
- `1` = not fully paid in the historical data

This is **not** a direct loan approval model. The app must be treated as an educational demonstration, not a real credit or lending decision tool.

## Features

- Overview with dataset statistics and model comparison
- Exploratory charts for purpose, FICO score, interest rate and numeric correlations
- Train/test evaluation of Random Forest, Extra Trees, Histogram Gradient Boosting and Logistic Regression
- Metrics: accuracy, precision, recall, F1 and ROC-AUC
- Hypothetical input form with model classification and estimated probability
- Dataset CSV download
- Original notebook preserved in `notebooks/` for reference

## Folder structure

```text
Loan_Repayment_Prediction/
├── app.py
├── requirements.txt
├── run_windows.bat
├── run_mac_linux.sh
├── data/
│   └── loan_data.csv
├── src/
│   └── modeling.py
└── notebooks/
    └── original_loan_repayment_notebook.ipynb
```

## Run on Windows

1. Install Python 3.10 or 3.11 from the official Python website and enable **Add Python to PATH** during setup.
2. Extract this ZIP to a normal folder (for example, `C:\Projects\Loan_Repayment_Prediction`).
3. Open that folder in VS Code, then choose **Terminal → New Terminal**.
4. Run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

5. Open the local address printed in the terminal, usually `http://localhost:8501`.

Alternatively, double-click `run_windows.bat` (it creates the environment and installs the packages on first run).

## Run on macOS / Linux

```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

## Dataset

The included CSV contains 9,578 records and 14 columns (13 features and the target). The data has a class imbalance: approximately 84% of records have `not.fully.paid = 0` and 16% have `not.fully.paid = 1`. For that reason, accuracy alone is not enough; check minority-class recall and F1 as well.

## Technical notes

- Preprocessing is inside scikit-learn pipelines, so imputation, scaling and category encoding are fitted on the training split only.
- The split is stratified, uses a fixed random seed, and reserves 25% for testing.
- `purpose` is one-hot encoded rather than treated as an ordinal number.
- The app trains models locally when needed; no applicant input is sent to an external API.

## Limitations

This is a classroom/demo project. Historical lending data can contain bias, missing context and outdated relationships. The model is not independently validated, calibrated or approved for real-world financial decisions. Avoid using it for actual loan eligibility, pricing or credit access.
