import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from shap.plots import _waterfall

# ========== 加载模型和标准化器 ==========
lr = joblib.load('logistic_regression_model.pkl')
vs1 = joblib.load('standardizer_vs1.pkl')

cb = CatBoostClassifier()
cb.load_model('catboost_model.cbm')
vs2 = joblib.load('standardizer_vs2.pkl')

# ========== 加载字段名 ==========
trainx1_columns = list(pd.read_csv('traindata1.csv', nrows=1).drop('PPCs', axis=1).columns)
trainx2_columns = list(pd.read_csv('traindata2.csv', nrows=1).drop('PPCs', axis=1).columns)

# ========== 加载最佳分类阈值 ==========
def load_thresholds(file="thresholds.csv"):
    df = pd.read_csv(file)
    return dict(zip(df['model'], df['threshold']))

thresholds = load_thresholds()
best_threshold_lr = thresholds.get("logistic_regression", 0.5)
best_threshold_cb = thresholds.get("catboost", 0.5)

# ========== Streamlit 页面配置 ==========
st.set_page_config(page_title="PPCs Prediction", layout="wide")
st.title("Prediction of Postoperative Pulmonary Complications (PPCs)")

model_type = st.sidebar.selectbox("Select model type:", ["Preoperative only", "Pre + Intraoperative"])

def preprocess_and_predict(inputs, model, vs, columns):
    df = pd.DataFrame([inputs], columns=columns)
    vs.transform(df)
    X = vs.standard_data_2()
    prob = model.predict_proba(X)[0, 1]
    return prob, X

# ========== 逻辑回归模型部分 ==========
if model_type == "Preoperative only":
    age = st.sidebar.number_input("Age", 18, 120)
    crp = st.sidebar.number_input("CRP (mg/L)", 0.01)
    diabetes = st.sidebar.selectbox("Diabetes mellitus", ['No', 'Yes'])
    smoking = st.sidebar.selectbox("Smoking", ['Never', 'Former', 'Current'])
    anemia = st.sidebar.selectbox("Preoperative anemia", ['No', 'Yes'])
    infection = st.sidebar.selectbox("Preoperative respiratory tract infection", ['No', 'Yes'])
    spo2 = st.sidebar.selectbox("Preoperative SpO2", ['≥96%', '<96%'])
    asa = st.sidebar.selectbox("ASA status", ['Ⅰ/Ⅱ', 'Ⅲ/Ⅳ/Ⅴ'])

    inputs = [
        age, crp,
        1 if diabetes == 'Yes' else 0,
        {'Never': 0, 'Former': 1, 'Current': 2}[smoking],
        1 if anemia == 'Yes' else 0,
        1 if infection == 'Yes' else 0,
        1 if spo2 == '<96%' else 0,
        1 if asa == 'Ⅲ/Ⅳ/Ⅴ' else 0,
    ]

    if st.button("Predict"):
        prob, X = preprocess_and_predict(inputs, lr, vs1, trainx1_columns)
        st.success(f"PPCs probability: {prob * 100:.1f}%")

        risk_group = "High risk" if prob >= best_threshold_lr else "Low risk"
        st.success(f"Risk group: {risk_group}")

        # SHAP解释
        explainer = shap.LinearExplainer(lr, vs1.standard_data_1(), feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots()
        _waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=trainx1_columns)
        st.pyplot(fig)

# ========== CatBoost 模型部分 ==========
else:
    age = st.sidebar.number_input("Age", 18, 120)
    crp = st.sidebar.number_input("CRP (mg/L)", 0.01)
    blood_loss = st.sidebar.number_input("Blood loss (mL)", 0)
    duration = st.sidebar.number_input("Surgery duration (min)", 0)
    diabetes = st.sidebar.selectbox("Diabetes mellitus", ['No', 'Yes'])
    stroke = st.sidebar.selectbox("Unexpected cerebrovascular disease", ['No', 'Yes'])
    smoking = st.sidebar.selectbox("Smoking", ['Never', 'Former', 'Current'])
    infection = st.sidebar.selectbox("Preoperative respiratory tract infection", ['No', 'Yes'])
    spo2 = st.sidebar.selectbox("Preoperative SpO2", ['≥96%', '<96%'])
    asa = st.sidebar.selectbox("ASA status", ['Ⅰ/Ⅱ', 'Ⅲ/Ⅳ/Ⅴ'])

    inputs = [
        age, crp, blood_loss, duration,
        1 if diabetes == 'Yes' else 0,
        1 if stroke == 'Yes' else 0,
        {'Never': 0, 'Former': 1, 'Current': 2}[smoking],
        1 if infection == 'Yes' else 0,
        1 if spo2 == '<96%' else 0,
        1 if asa == 'Ⅲ/Ⅳ/Ⅴ' else 0,
    ]

    if st.button("Predict"):
        prob, X = preprocess_and_predict(inputs, cb, vs2, trainx2_columns)
        st.success(f"PPCs probability: {prob * 100:.1f}%")

        risk_group = "High risk" if prob >= best_threshold_cb else "Low risk"
        st.success(f"Risk group: {risk_group}")

        # SHAP解释
        explainer = shap.TreeExplainer(cb)
        shap_values = explainer(X)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

# ========== 页面底部 ==========
st.markdown("---")
st.caption("2023 Nanjing First Hospital. Contact: zoujianjun100@126.com")
