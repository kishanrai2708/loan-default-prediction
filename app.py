import json
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("xgb_model.json")
with open("meta.json") as f:
    art = json.load(f)
cols, caps, thr = art["columns"], art["caps"], art["threshold"]

st.set_page_config(page_title="Loan Default Risk", page_icon="🏦")
st.title("Loan Default Risk Estimator")
st.caption("Demo built on the 'Give Me Some Credit' dataset. For learning only, not financial advice.")

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 110, 40)
    income_known = st.checkbox("Monthly income is known", value=True)
    income = st.number_input("Monthly income", 0, 1_000_000, 5000, disabled=not income_known)
    dependents = st.number_input("Number of dependents", 0, 20, 0)
    open_lines = st.number_input("Open credit lines and loans", 0, 60, 8)
    real_estate = st.number_input("Real estate loans", 0, 60, 1)
with c2:
    util = st.number_input("Credit utilization (balance / limit)", 0.0, 10.0, 0.3, step=0.05)
    debt_ratio = st.number_input("Debt ratio (monthly debt / income)", 0.0, 5000.0, 0.4, step=0.05)
    late30 = st.number_input("Times 30-59 days late", 0, 20, 0)
    late60 = st.number_input("Times 60-89 days late", 0, 20, 0)
    late90 = st.number_input("Times 90+ days late", 0, 20, 0)

if st.button("Estimate risk", type="primary"):
    row = {
        "RevolvingUtilizationOfUnsecuredLines": min(util, caps["RevolvingUtilizationOfUnsecuredLines"]),
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late30,
        "DebtRatio": min(debt_ratio, caps["DebtRatio"]),
        "MonthlyIncome": min(income if income_known else art["income_median"], caps["MonthlyIncome"]),
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late60,
        "NumberOfDependents": dependents,
        "late_code_flag": 0,
        "income_missing": 0 if income_known else 1,
    }
    X = pd.DataFrame([row])[cols]
    score = float(model.predict_proba(X)[:, 1][0])

    if score >= thr:
        band = "High risk"
        st.error(band)
    elif score >= thr / 2:
        band = "Medium risk"
        st.warning(band)
    else:
        band = "Low risk"
        st.success(band)

    st.metric("Risk score", f"{score:.2f}")
    st.caption(f"The score ranks borrowers by risk; it is not a literal probability. "
               f"Scores of {thr:.2f} or higher are flagged as high risk.")