from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from modeling import load_data, evaluate_models, build_pipeline, FEATURES, TARGET, PURPOSES

st.set_page_config(page_title="LoanLens | Repayment Risk", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px;}
[data-testid="stMetric"] {background: linear-gradient(135deg, rgba(72,99,255,.12), rgba(31,184,170,.10)); padding: 16px 18px; border: 1px solid rgba(120,140,180,.18); border-radius: 16px;}
.hero {padding: 26px 30px; border-radius: 20px; background: linear-gradient(120deg,#111c3b,#263d70 58%,#177e89); color: white; margin-bottom: 22px;}
.hero h1 {font-size: 2.25rem; margin: 0 0 8px 0; color: white;} .hero p {font-size: 1.02rem; opacity: .9; margin: 0;}
.small-muted {color: #78839a; font-size: .9rem;}
</style>
<div class="hero"><h1>LoanLens</h1><p>Loan repayment risk explorer · Machine learning powered analysis of historical lending data</p></div>
""", unsafe_allow_html=True)



@st.cache_data(show_spinner=False)
def get_data():
    return load_data(ROOT / "data" / "loan_data.csv")

@st.cache_resource(show_spinner="Training and evaluating models on the included dataset...")
def get_evaluation():
    return evaluate_models(get_data())

df = get_data()
results, fitted, split_data = get_evaluation()

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio("Choose a page", ["Overview", "Explore the data", "Predict repayment risk", "Model performance"], label_visibility="collapsed")
    st.divider()
    st.caption("Project: Loan Repayment Prediction")
    st.caption(f"Dataset: {len(df):,} historical records")
    st.caption("Local demo · no applicant data is uploaded")

if page == "Overview":
    st.subheader("Project overview")
    st.write("This application explores whether a loan was **not fully paid** in the historical dataset. It provides descriptive analytics, compares several baseline classifiers, and demonstrates a model-based risk estimate for hypothetical inputs.")
    target_rate = df[TARGET].mean() * 100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Loan records", f"{len(df):,}")
    c2.metric("Input features", f"{len(FEATURES)}")
    c3.metric("Historical not-fully-paid rate", f"{target_rate:.1f}%")
    c4.metric("Best test ROC-AUC", f"{results['ROC-AUC'].max():.3f}")
    st.info("Important: this is an educational ML demonstration, not a credit decision system. The historical target is `not.fully.paid`; it is not a direct prediction of whether a new loan should be approved. Do not use the output for real lending decisions.")
    left,right = st.columns([1.2,1])
    with left:
        st.markdown("### Target distribution")
        counts = df[TARGET].map({0:"Fully paid",1:"Not fully paid"}).value_counts().rename_axis("Outcome").reset_index(name="Count")
        fig = px.pie(counts, names="Outcome", values="Count", hole=.55, color_discrete_sequence=["#4563eb","#14a89a"])
        fig.update_layout(margin=dict(l=5,r=5,t=10,b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Model comparison snapshot")
        view = results.sort_values("ROC-AUC", ascending=False)[["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]]
        st.dataframe(view.style.format({c:"{:.3f}" for c in view.columns if c != "Model"}), use_container_width=True, hide_index=True)
    st.markdown("### Included workflow")
    st.markdown("1. Load and validate the provided CSV.\n2. Explore the distribution and relationships in the data.\n3. Train multiple models using a stratified train/test split.\n4. Compare accuracy alongside precision, recall, F1 and ROC-AUC.\n5. Enter a hypothetical applicant profile to see an illustrative model estimate.")

elif page == "Explore the data":
    st.subheader("Explore the data")
    st.write("Use these charts to inspect patterns in the dataset. Associations are descriptive and do not establish causation.")
    c1,c2 = st.columns(2)
    with c1:
        purpose = df.groupby(["purpose", TARGET]).size().reset_index(name="Loans")
        purpose["Outcome"] = purpose[TARGET].map({0:"Fully paid",1:"Not fully paid"})
        fig = px.bar(purpose, x="purpose", y="Loans", color="Outcome", barmode="group", title="Loan purpose by repayment outcome", color_discrete_sequence=["#4563eb","#f29e4c"])
        fig.update_layout(xaxis_title="Loan purpose", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="fico", color=df[TARGET].map({0:"Fully paid",1:"Not fully paid"}), barmode="overlay", opacity=.7, nbins=35, title="FICO score distribution", color_discrete_sequence=["#4563eb","#f29e4c"])
        fig.update_layout(xaxis_title="FICO score", legend_title_text="Outcome")
        st.plotly_chart(fig, use_container_width=True)
    c3,c4 = st.columns(2)
    with c3:
        fig = px.box(df, x=df[TARGET].map({0:"Fully paid",1:"Not fully paid"}), y="int.rate", color=df[TARGET].map({0:"Fully paid",1:"Not fully paid"}), title="Interest rate by outcome", color_discrete_sequence=["#4563eb","#f29e4c"])
        fig.update_layout(xaxis_title="Repayment outcome", yaxis_title="Interest rate")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric feature correlation")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Dataset preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download dataset CSV", df.to_csv(index=False).encode("utf-8"), "loan_data.csv", "text/csv")

elif page == "Predict repayment risk":
    st.subheader("Hypothetical applicant estimate")
    st.write("Enter values in the form below. Fields and ranges are based on the included historical dataset. The estimate is not a loan approval or rejection recommendation.")
    model_name = st.selectbox("Model", list(fitted.keys()), index=list(fitted.keys()).index("Random Forest") if "Random Forest" in fitted else 0)
    numeric = df[FEATURES].select_dtypes(include=np.number)
    with st.form("prediction_form"):
        a,b,c = st.columns(3)
        with a:
            credit_policy = st.selectbox("Credit policy met?", [1,0], format_func=lambda x: "Yes" if x == 1 else "No", help="Dataset indicator; not a bank approval decision.")
            purpose = st.selectbox("Loan purpose", PURPOSES)
            int_rate = st.slider("Interest rate", 0.06, 0.22, 0.12, 0.001, format="%.3f")
            installment = st.number_input("Monthly installment ($)", min_value=15.0, max_value=2000.0, value=320.0, step=10.0)
        with b:
            log_income = st.slider("Log annual income", 7.5, 14.6, 10.9, 0.05)
            dti = st.slider("Debt-to-income ratio", 0.0, 30.0, 12.6, 0.1)
            fico = st.slider("FICO score", 612, 827, 711)
            days_line = st.number_input("Days with credit line", min_value=100.0, max_value=18000.0, value=4560.0, step=100.0)
        with c:
            revol_bal = st.number_input("Revolving balance ($)", min_value=0, max_value=1500000, value=16900, step=500)
            revol_util = st.slider("Revolving utilization (%)", 0.0, 119.0, 46.8, 0.5)
            inquiries = st.number_input("Credit inquiries in last 6 months", min_value=0, max_value=35, value=1, step=1)
            delinquencies = st.number_input("Delinquencies in last 2 years", min_value=0, max_value=20, value=0, step=1)
            public_records = st.number_input("Public records", min_value=0, max_value=10, value=0, step=1)
        submitted = st.form_submit_button("Estimate historical repayment risk", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{
            "credit.policy": int(credit_policy), "purpose": purpose, "int.rate": float(int_rate), "installment": float(installment),
            "log.annual.inc": float(log_income), "dti": float(dti), "fico": int(fico), "days.with.cr.line": float(days_line),
            "revol.bal": int(revol_bal), "revol.util": float(revol_util), "inq.last.6mths": int(inquiries),
            "delinq.2yrs": int(delinquencies), "pub.rec": int(public_records)
        }])[FEATURES]
        model = fitted[model_name]
        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
        label = "Not fully paid" if pred == 1 else "Fully paid"
        x,y = st.columns([1,2])
        with x:
            st.metric("Model classification", label)
            if proba is not None: st.metric("Estimated not-fully-paid probability", f"{proba:.1%}")
        with y:
            if proba is not None:
                gauge = go.Figure(go.Indicator(mode="gauge+number", value=proba*100, number={"suffix":"%"}, title={"text":"Model-estimated probability"}, gauge={"axis":{"range":[0,100]},"bar":{"color":"#4563eb"},"steps":[{"range":[0,30],"color":"#d9f4eb"},{"range":[30,60],"color":"#fff0c2"},{"range":[60,100],"color":"#ffd9d9"}]}))
                gauge.update_layout(height=260, margin=dict(l=20,r=20,t=45,b=5))
                st.plotly_chart(gauge, use_container_width=True)
        st.warning("This result is an educational estimate from historical data. It may be biased, poorly calibrated, and unsuitable for real-world financial decisions. It must not be used as the sole basis for lending, pricing, or eligibility.")

else:
    st.subheader("Model performance")
    st.write("Models use a stratified 75/25 train/test split with a fixed random seed. Preprocessing is fitted inside each pipeline to avoid leakage from the test split.")
    display = results.sort_values("ROC-AUC", ascending=False)
    st.dataframe(display.style.format({c:"{:.3f}" for c in display.columns if c != "Model"}), use_container_width=True, hide_index=True)
    fig = px.bar(display.melt(id_vars="Model", value_vars=["Precision","Recall","F1","ROC-AUC"], var_name="Metric", value_name="Score"), x="Model", y="Score", color="Metric", barmode="group", title="Test-set metrics by model", range_y=[0,1])
    st.plotly_chart(fig, use_container_width=True)
    model_name = st.selectbox("Inspect confusion matrix", list(fitted.keys()))
    X_test, y_test = split_data["X_test"], split_data["y_test"]
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
    pred = fitted[model_name].predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    fig = px.imshow(cm, text_auto=True, x=["Predicted fully paid","Predicted not fully paid"], y=["Actual fully paid","Actual not fully paid"], color_continuous_scale="Blues", title=f"Confusion matrix — {model_name}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Because the target classes are imbalanced, accuracy alone can be misleading. Compare recall and F1 for the not-fully-paid class as well.")

st.divider()
st.caption("LoanLens · Student ML project · Uses the provided LendingClub-style loan dataset. No data is sent to an external service.")
