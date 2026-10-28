# Analysis of supplied materials

- The supplied notebook is titled **Loan Repayment Prediction Using Ensemble Learning Methods**.
- The supplied CSV has **9,578 rows and 14 columns**.
- Target column: `not.fully.paid` (0 = fully paid, 1 = not fully paid).
- Class counts: {0: 8045, 1: 1533} (imbalanced classes).
- The original notebook reports roughly 85% accuracy for tree ensemble models, but the reported gradient-boosting results show very low recall for the not-fully-paid class. Therefore the upgraded app reports precision, recall, F1 and ROC-AUC alongside accuracy.
- The project is implemented as a local Streamlit web app with exploratory data analysis, model comparison, and a hypothetical prediction form.
- The app is for education only and must not be used for real lending decisions.
