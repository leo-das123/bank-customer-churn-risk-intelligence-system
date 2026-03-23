import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import seaborn as sns

# =========================================
# Page Config
# =========================================

st.set_page_config(
page_title="Bank Churn Intelligence",
layout="wide"
)

# Dark professional styling
st.markdown("""
<style>

.stApp{
background-color:#0E1117;
color:white;
}

.metric-card{
background:#1C1F26;
padding:20px;
border-radius:12px;
box-shadow:0px 4px 15px rgba(0,0,0,.4);
text-align:center;
}

</style>
""",unsafe_allow_html=True)

# =========================================
# Load Model
# =========================================

saved = joblib.load("churn_model.pkl")

model = saved["model"]
threshold = saved["threshold"]

X_test = saved.get("X_test")
y_test = saved.get("y_test")

# =========================================
# Header
# =========================================

st.title("🏦 AI Customer Churn Intelligence Platform")

st.markdown("""
Enterprise decision support system for churn prediction and retention strategy.
""")

# =========================================
# Sidebar
# =========================================

st.sidebar.header("Customer Profile")

credit_score = st.sidebar.slider("Credit Score",300,900,600)

age = st.sidebar.slider("Age",18,100,35)

tenure = st.sidebar.slider("Tenure",0,10,3)

balance = st.sidebar.number_input(
"Account Balance",
value=50000.0
)

num_products = st.sidebar.slider(
"Products Used",
1,
4,
1
)

has_card_ui = st.sidebar.selectbox(
"Credit Card",
["Yes","No"]
)

is_active_ui = st.sidebar.selectbox(
"Active Member",
["Yes","No"]
)

salary = st.sidebar.number_input(
"Estimated Salary",
value=50000.0
)

geography = st.sidebar.selectbox(
"Geography",
[  "France","Spain","Germany"]
)

gender = st.sidebar.selectbox(
"Gender",
["Male","Female"]
)

has_card = 1 if has_card_ui=="Yes" else 0
is_active = 1 if is_active_ui=="Yes" else 0

# =========================================
# Feature Engineering
# =========================================

balance_salary_ratio = balance/(salary+1)

age_tenure_ratio = age/(tenure+1)

product_per_year = num_products/(tenure+1)

input_data = pd.DataFrame({

"CreditScore":[credit_score],
"Geography":[geography],
"Gender":[gender],
"Age":[age],
"Tenure":[tenure],
"Balance":[balance],
"NumOfProducts":[num_products],
"HasCrCard":[has_card],
"IsActiveMember":[is_active],
"EstimatedSalary":[salary],
"BalanceSalaryRatio":[balance_salary_ratio],
"AgeTenureRatio":[age_tenure_ratio],
"ProductPerYear":[product_per_year]

})

# =========================================
# Analyze Button
# =========================================

analyze = st.button("Analyze Customer")

if analyze:

    probability = model.predict_proba(input_data)[0][1]

    risk_score = int(probability*100)

# =========================================
# KPI CARDS
# =========================================

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    col2.metric(
        "Risk Score",
        f"{risk_score}/100"
    )

    if probability<.30:

        col3.success("LOW RISK")

    elif probability<.60:

        col3.warning("MEDIUM RISK")

    else:

        col3.error("HIGH RISK")

# =========================================
# Risk Gauge
# =========================================

    st.subheader("Risk Gauge")

    fig,ax=plt.subplots(figsize=(8,2))

    ax.barh(
    ["Risk"],
    [risk_score]
    )

    ax.set_xlim(0,100)

    st.pyplot(fig)

# =========================================
# Probability Distribution
# =========================================

    st.subheader("Risk vs Population")

    fig2,ax2=plt.subplots()

    ax2.hist(
    model.predict_proba(X_test)[:,1],
    bins=30
    )

    ax2.axvline(
    probability,
    color="red"
    )

    st.pyplot(fig2)

# =========================================
# Feature Importance
# =========================================

    st.subheader("Key Risk Drivers")

    rf_model=model.named_steps["classifier"]

    importances=rf_model.feature_importances_

    feature_names=model.named_steps[
    "preprocessor"
    ].get_feature_names_out()

    importance_df=pd.DataFrame({

    "Feature":feature_names,
    "Importance":importances

    }).sort_values(
    "Importance",
    ascending=False
    ).head(8)

    fig3,ax3=plt.subplots()

    ax3.barh(
    importance_df["Feature"],
    importance_df["Importance"]
    )

    ax3.invert_yaxis()

    st.pyplot(fig3)

# =========================================
# Retention Engine
# =========================================

    st.subheader("Retention Strategy")

    if probability>.60:

        st.error("""
Recommended Actions:

• Immediate RM contact
• Loyalty incentive
• Product bundle offer
• Service upgrade
""")

    elif probability>.30:

        st.warning("""
Recommended Actions:

• Monitor engagement
• Offer service benefits
• Improve interaction
""")

    else:

        st.success("""
Recommended Actions:

• Maintain relationship
• Offer rewards
""")

# =========================================
# Executive Decision
# =========================================

    st.subheader("Executive Decision")

    if probability>.60:

        st.write("Decision: Immediate retention campaign")

    elif probability>.30:

        st.write("Decision: Monitor behavior")

    else:

        st.write("Decision: Maintain relationship")

# =========================================
# Download Report
# =========================================

    report=pd.DataFrame({

    "Risk Score":[risk_score],
    "Probability":[probability]

    })

    st.download_button(

    "Download Risk Report",

    report.to_csv(),

    file_name="customer_risk.csv"

    )

# =========================================
# Batch Scoring (FIXED)
# =========================================

st.markdown("---")

st.subheader("Batch Customer Scoring")

file = st.file_uploader(
"Upload CSV",
type=["csv"]
)

if file:

    batch = pd.read_csv(file)

    # Ensure required columns exist
    required = [
    "Balance",
    "EstimatedSalary",
    "Age",
    "Tenure",
    "NumOfProducts"
    ]

    if not all(col in batch.columns for col in required):

        st.error("CSV missing required columns")

    else:

        # Feature engineering (same as training)

        batch["BalanceSalaryRatio"] = (
        batch["Balance"] /
        (batch["EstimatedSalary"] + 1)
        )

        batch["AgeTenureRatio"] = (
        batch["Age"] /
        (batch["Tenure"] + 1)
        )

        batch["ProductPerYear"] = (
        batch["NumOfProducts"] /
        (batch["Tenure"] + 1)
        )

        # Predict
        probs = model.predict_proba(batch)[:,1]

        batch["RiskProbability"] = probs

        batch["RiskScore"] = (
        probs * 100
        ).astype(int)

        st.dataframe(batch.head())

        st.download_button(

        "Download Results",

        batch.to_csv(index=False),

        file_name="batch_results.csv"

        )

# =========================================
# Hidden Technical Panel
# =========================================

with st.expander("Model Diagnostics"):

    if X_test is not None:

        y_prob=model.predict_proba(X_test)[:,1]

        y_pred=(y_prob>=threshold).astype(int)

        st.write(
        "ROC-AUC:",
        round(roc_auc_score(y_test,y_prob),3)
        )

        cm=confusion_matrix(y_test,y_pred)

        fig4,ax4=plt.subplots()

        sns.heatmap(cm,annot=True)

        st.pyplot(fig4)