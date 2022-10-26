from base64 import a85decode
from copyreg import pickle
import streamlit as st
import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import pandas as pd
import shap
import xgboost as xgb  ###xgboost
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt



st.set_page_config(page_title="Probability prediction of pulmonary complications after Non-cardiothoracic Surgery", layout="wide")

plt.style.use('default')

df=pd.read_csv('traindata1.csv',encoding='utf8')
trainy=df.PPCs
trainx=df.drop('PPCs',axis=1)
xgb = XGBClassifier(colsample_bytree=1,gamma=1,learning_rate=0.01,max_depth=2,
                    n_estimators =400,min_child_weight=1,subsample=0.5,
                    objective= 'binary:logistic',random_state = 1)
xgb.fit(trainx,trainy)

###side-bar
def user_input_features():
    st.title("Probability prediction of pulmonary Complications")
    st.sidebar.header('User input parameters below')
    a1=st.sidebar.number_input("Age",min_value=18,max_value=120)
    a2=st.sidebar.number_input("CRP(mg/L)",min_value=0.01,max_value=None,step=0.01)
    a3=st.sidebar.number_input("Surgery duration(min)",min_value=0,max_value=None)
    a4=st.sidebar.number_input("Intraoperative blood loss(ml)",min_value=0,max_value=None)
    a5=st.sidebar.selectbox('Cerebrovascular accident',('No','Yes'))
    a6=st.sidebar.selectbox('Preoperative anemia',('No','Yes'))
    a7=st.sidebar.selectbox('Respiratory infection within 30 days',('No','Yes'))
    a8=st.sidebar.selectbox('Preoperative SpO2',('≥96','<96'))
    result=""
    if a5=="Yes":
        a5=1
    else: 
        a5=0 
    if a6=="Yes":
        a6=1
    else: 
        a6=0 
    if a7=="Yes":
        a7=1
    else: 
        a7=0 
    if a8=="≥96":
        a8=0
    else: 
        a8=1 
    output=[a1,a2,a3,a4,a5,a6,a7,a8]
    patient1=pd.DataFrame(output)
    patient=pd.DataFrame(patient1.values.T,columns=trainx.columns)
    prediction=xgb.predict_proba(patient)
    prediction=float(prediction[:, 1])
    def predict_PPCs():
        prediction=round(user_input_features[:, 1],3)
        return prediction
    result=""
    if st.button("Predict"):
        st.success('The probability of PPCs for the patient: {:.1f}%'.format(prediction*100))
        if prediction>0.148:
            b="High risk"
        else:
            b="Low risk"
        st.success('The risk group:'+ b)
        explainer_xgb = shap.TreeExplainer(xgb)
        shap_values= explainer_xgb(patient)
        shap.plots.waterfall(shap_values[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Waterfall plot analysis of PPCs for the patient:")
        st.pyplot(bbox_inches='tight')


        st.write("Abbreviations:PPCs,pulmonary Complications; CRP, C-reactive protein; Preoperative SpO2, Preoperative oxygen saturation")
    if st.button("Reset"):
        st.write("")
    st.markdown("*Statement: this website will not record or store any information inputed.")
    st.write("2022 Nanjing First Hospital, Nanjing Medical University. All Rights Reserved ")
    st.write("✉ Contact Us: zoujianjun100@126.com")

if __name__ == '__main__':
    user_input_features()
