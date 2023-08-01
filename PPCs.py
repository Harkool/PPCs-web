from base64 import a85decode
from copyreg import pickle
import streamlit as st
import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import pandas as pd
import shap
from IPython.display import display 
from sklearn.linear_model import LogisticRegression as LR
import catboost as cb
from catboost.datasets import titanic
import plotly.graph_objs as go 
import plotly.express as px
import matplotlib.pyplot as plt
from shap.plots import _waterfall
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler



st.set_page_config(page_title="Prediction probability of postoperative pulmonary complications after non-cardiothoracic surgery", layout="wide")
plt.style.use('default')
df1=pd.read_csv('traindata1.csv',encoding='utf8')
df2=pd.read_csv('traindata2.csv',encoding='utf8')
trainy1=df1.PPCs
trainx1=df1.drop('PPCs',axis=1)
trainy2=df2.PPCs
trainx2=df2.drop('PPCs',axis=1)

lr= LR(C=11,penalty="l2", 
       solver = "liblinear",
       class_weight = {0:0.85,1:0.15},
       random_state = 1)
lr=lr.fit(trainx1, trainy1)



Cb = cb.CatBoostClassifier(
         iterations=200,
        od_type='Iter',
        od_wait=600,
        max_depth=5,
        learning_rate=0.01,
        l2_leaf_reg=12,
        random_seed=1,
        metric_period=50,
        fold_len_multiplier=1.2,
        loss_function='Logloss',
        logging_level='Verbose')
Cb.fit(trainx2, trainy2)

###side-bar
def user_input_features():
    st.title("Prediction probability of postoperative pulmonary complications")
    st.sidebar.header('User input parameters below')
    a0=st.sidebar.selectbox('Type of model',('Models with only preoperative variables','Models with preoperative and intraoperative variables'))
    if a0=="Models with only preoperative variables":
        a1=st.sidebar.number_input("Age",min_value=18,max_value=120)
        a2=st.sidebar.number_input("CRP(mg/L)",min_value=0.01,max_value=None,step=0.01)
        a3=st.sidebar.selectbox('Diabetes mellitus',('No','Yes'))
        a4=st.sidebar.selectbox('Smoking',('Never smoking','Former smoking','Current smoking'))
        a5=st.sidebar.selectbox('Preoperative anemia',('No','Yes'))
        a6=st.sidebar.selectbox('Preoperative respiratory tract infection',('No','Yes'))
        a7=st.sidebar.selectbox('Preoperative SpO2',('≥96%','<96%'))
        a8=st.sidebar.selectbox('ASA physical status',('Ⅰ/Ⅱ','Ⅲ/Ⅳ/Ⅴ'))
        if a3=="No":
            a3=0
        else:
            a3=1
        if a4=="Never smoking":
            a4=0
        elif a4=='Former smoking':
            a4=1
        else: 
            a4=2 
        if a5=="No":
            a5=0
        else:
            a5=1
        if a6=="No":
            a6=0
        else:
            a6=1
        if a7=="<96%":
            a7=1
        else: 
            a7=0 
        if a8=="Ⅲ/Ⅳ/Ⅴ":
            a8=1
        else: 
            a8=0 
   
        output=[a1,a2,a3,a4,a5,a6,a7,a8]
        int_features=[int(x) for x in output]
        final_features=np.array(int_features)
        patient1=pd.DataFrame(final_features)
        patient=pd.DataFrame(patient1.values.T,columns=trainx1.columns)

        prediction=lr.predict_proba(patient)
        prediction=float(prediction[:, 1])
        def predict_PPCs():
            prediction=round(user_input_features[:, 1],3)
            return prediction
        result=""
        if st.button("Predict"):
            st.success('The probability of PPCs for the patient: {:.1f}%'.format(prediction*100))
            if prediction>0.047:
                b="High risk"
            else:
                b="Low risk"
            st.success('The risk group: '+ b)

            x_train_summary = shap.kmeans(trainx1,1)
            explainer_lr=shap.KernelExplainer(lr.predict_proba,x_train_summary, feature_names=trainx1.columns)
            shap_values= explainer_lr.shap_values(patient)


        #explainer_Cb = shap.TreeExplainer(Cb)
        #shap_values= explainer_Cb(patient)
            _waterfall.waterfall_legacy(explainer_lr.expected_value[0], shap_values[1][0], feature_names=trainx1.columns)
        #shap.plots.waterfall(shap_values[0])
        #shap.plots.force(shap_values[0])

            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write("Waterfall plot analysis of PPCs for the patient:")
            st.pyplot(bbox_inches='tight')
            st.write("Abbreviations: PPCs, postoperative pulmonary complications; CRP, C-reactive protein; SpO2, Peripheral capillary oxygen saturation; ASA, American Society of Anesthesiologists.")
        if st.button("Reset"):
            st.write("")
        st.markdown("*Statement: this website will not record or store any information inputed.")
        st.write("2023 Nanjing First Hospital, Nanjing Medical University. All Rights Reserved ")
        st.write("✉ Contact Us: zoujianjun100@126.com")
    else:
            a1=st.sidebar.number_input("Age",min_value=18,max_value=120)
            a2=st.sidebar.number_input("CRP(mg/L)",min_value=0.01,max_value=None,step=0.01)
            a3=st.sidebar.number_input("Intraoperative blood loss(mL)",min_value=0,max_value=None,step=1)
            a4=st.sidebar.number_input("Surgery duration (min)",min_value=0,max_value=None,step=1)
            a5=st.sidebar.selectbox('Diabetes mellitus',('No','Yes'))
            a6=st.sidebar.selectbox('Unexpected cerebrovascular disease',('No','Yes'))
            a7=st.sidebar.selectbox('Smoking',('Never smoking','Former smoking','Current smoking'))
            a8=st.sidebar.selectbox('Preoperative respiratory tract infection',('No','Yes'))
            a9=st.sidebar.selectbox('Preoperative SpO2',('≥96%','<96%'))
            a10=st.sidebar.selectbox('ASA physical status',('Ⅰ/Ⅱ','Ⅲ/Ⅳ/Ⅴ'))
            if a5=="No":
                a5=0
            else:
                a5=1
            if a6=="No":
                a6=0
            else:
                a6=1
            if a7=="Never smoking":
                a7=0
            elif a7=='Former smoking':
                a7=1
            else: 
                a7=2 
            if a8=="No":
                a8=0
            else:
                a8=1
            if a9=="<96%":
                a9=1
            else: 
                a9=0 
            if a10=="Ⅲ/Ⅳ/Ⅴ":
                a10=1
            else: 
                a10=0 
   
            output=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
            int_features=[int(x) for x in output]
            final_features=np.array(int_features)
            patient1=pd.DataFrame(final_features)
            patient=pd.DataFrame(patient1.values.T,columns=trainx2.columns)

            prediction=Cb.predict_proba(patient)
            prediction=float(prediction[:, 1])
            def predict_PPCs():
                prediction=round(user_input_features[:, 1],3)
                return prediction
            result=""
            if st.button("Predict"):
                st.success('The probability of PPCs for the patient: {:.1f}%'.format(prediction*100))
                if prediction>0.047:
                    b="High risk"
                else:
                    b="Low risk"
                st.success('The risk group: '+ b)
                explainer_Cb = shap.TreeExplainer(Cb)
                
                shap_values= explainer_Cb(patient)
                shap.plots.waterfall(shap_values[0])



                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.write("Waterfall plot analysis of PPCs for the patient:")
                st.pyplot(bbox_inches='tight')
                st.write("Abbreviations: PPCs, postoperative pulmonary complications; CRP, C-reactive protein; SpO2, Peripheral capillary oxygen saturation; ASA, American Society of Anesthesiologists.")
            if st.button("Reset"):
                st.write("")
            st.markdown("*Statement: this website will not record or store any information inputed.")
            st.write("2023 Nanjing First Hospital, Nanjing Medical University. All Rights Reserved ")
            st.write("✉ Contact Us: zoujianjun100@126.com")

if __name__ == '__main__':
    user_input_features()
