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

class Valuestandard:
    
    def __init__(self, skip_list=None, onehot_list=None, categories_list=None, Skip=False, OneHot=False):
        
        self.__Skip = Skip
        self.__OneHot = OneHot
        
        if self.__Skip:
            if type(skip_list) == type(None):
                raise ValueError("需要参数：skip_list")
            elif type(skip_list) != type([]):
                raise ValueError("参数“skip_list”的类型必须为：list")
            else:
                self.__list1 = skip_list
        
        if self.__OneHot:
            if type(onehot_list) == type(None):
                raise ValueError("需要参数：onehot_list")
            elif type(onehot_list) != type([]):
                raise ValueError("参数“onehot_list”的类型必须为：list")
            else:
                self.__list2 = onehot_list
        
        if self.__OneHot:
            if type(categories_list) == type(None):
                raise ValueError("需要参数：categories_list")
            elif type(categories_list) != type([]):
                raise ValueError("参数“categories_list”的类型必须为：list")
            else:
                for self.__k in range(len(categories_list)):
                    if self.__k == 0:
                        self.__list3 = [list(range(categories_list[self.__k]))]
                    else:
                        self.__list3.append(list(range(categories_list[self.__k])))
                
        self.__fitstate = False
        self.__fitstate_2 = False
        
        return None
    
    def fit_transform(self, data_1=None):
        """
        fit_transform 方法用于标准化数据
        
        参数介绍
        ========
        data_1：待标准化的数据集，类型为DataFrame，默认为None
        ========
        """
        
        if self.__fitstate:
            raise RuntimeError("数据已被标准化，请勿重复标准化；若要重新标准化，请重新实例化Valuestandard")
        
        if type(data_1) == None:
            raise ValueError("需要待标准化的数据集：data_1")
        elif type(data_1) != type(pd.DataFrame()):
            raise ValueError("数据集的类型必须为：DataFrame，而传入数据的类型为：%s" % type(data_1))
        else:
            self.__Data_1 = data_1.copy()
            self.__data = self.__Data_1.copy()
            
        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)
            
            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)
                
                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)
                
                self.__transdata = None
                self.__quantity = None
                self.__OH = None
                self.__LE = None
                
            self.__Oname_list = None
            self.__data_1 = None
        
        if self.__Skip:
            self.__Sname_list = self.__Data_1.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            if self.__data.shape[1] != 0:
                self.__s = True
            else:
                self.__s = False
            
            self.__Sname_list = None
        else:
            self.__s = True 
            
        if self.__s:
            self.__Dname_list = self.__data.columns
            self.__data_original = self.__data.copy()
            self.__Key = StandardScaler()
            self.__data_standard = self.__Key.fit_transform(self.__data)
            self.__data_standard = pd.DataFrame(self.__data_standard)
            self.__data_standard.columns = self.__Dname_list
            self.__Dname_list = None
        else:
            self.__data_standard = self.__data.copy()
            self.__data_original = self.__data.copy()
        
        if self.__Skip:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_skip], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_o2], axis=1)
            self.__data_skip = None
            self.__data_o2 = None
        
        if self.__OneHot:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_onehot], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_onehot], axis=1)
            self.__data_onehot = None
        
        self.__data = None
        self.__fitstate = True
        
        return None
    
    def transform(self, data_2=None):
        """
        transform 方法用于，以fit_transform的标准，标准化数据，如测试集数据等
        
        参数介绍
        ========
        data_2：待标准化的数据集，类型为DataFrame，默认为None
        
        ★可以调用多次，但只保存最后一次调用的数据★
        ========
        """
        if self.__fitstate != True:
            raise RuntimeError("缺少数据标准化标准，请先调用 fit_transform() 方法")
        
        if type(data_2) == None:
            raise ValueError("需要待标准化的数据集：data_2")
        elif type(data_2) != type(pd.DataFrame()):
            raise ValueError("数据集的类型必须为：DataFrame，而传入数据的类型为：%s" % type(data_2))
        else:
            self.__Data_2 = data_2.copy()
            self.__data = self.__Data_2.copy()
            
        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)
            
            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)
                
                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)
                
                self.__transdata = None
                self.__quantity = None
                self.__OH = None
                self.__LE = None
                
            self.__Oname_list = None
            self.__data_1 = None
        
        if self.__Skip:
            self.__Sname_list = self.__Data_2.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            if self.__data.shape[1] != 0:
                self.__st = True
            else:
                self.__st = False
            
            self.__Sname_list = None
        else:
            self.__st = True
            
        if self.__st:
            self.__Dname_list = self.__data.columns
            self.__data_original_2 = self.__data.copy()
            self.__data_standard_2 = self.__Key.transform(self.__data)
            self.__data_standard_2 = pd.DataFrame(self.__data_standard_2)
            self.__data_standard_2.columns = self.__Dname_list
            self.__Dname_list = None
        else:
            self.__data_standard_2 = self.__data.copy()
            self.__data_original_2 = self.__data.copy()
        
        if self.__Skip:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_skip], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_o2], axis=1)
            self.__data_skip = None
            self.__data_o2 = None
        
        if self.__OneHot:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_onehot], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_onehot], axis=1)
            self.__data_onehot = None
        
        self.__data = None
        self.__fitstate_2 = True
        
        return None
        
    
    def standard_data_1(self):
        """
        standard_data_1 用于返回fit_transform()方法标准化后的数据，需先调用fit_transform()方法
        """
        if self.__fitstate != True:
            raise RuntimeError("没有数据被标准化，请先调用 fit_transform() 方法")
        
        return self.__data_standard
    
    def standard_data_2(self):
        """
        standard_data_2 用于返回transform()方法标准化后的数据，需先调用transform()方法
        """
        if self.__fitstate_2 != True:
            raise RuntimeError("没有数据被标准化，请先调用 transform() 方法")
        
        return self.__data_standard_2
    
    def original_data_1(self):
        """
        用于返回fit_transform()方法标准化后的原始数据，需先调用fit_transform()方法
        """
        if self.__fitstate != True:
            raise RuntimeError("请先调用 fit_transform() 方法")
        
        return self.__data_original
    
    def original_data_2(self):
        """
        用于返回transform方法标准化后的原始数据，需先调用transform方法
        """
        if self.__fitstate_2 != True:
            raise RuntimeError("请先调用 transform() 方法")
            
        return self.__data_original_2


st.set_page_config(page_title="Prediction probability of postoperative pulmonary complications after non-cardiothoracic surgery", layout="wide")
plt.style.use('default')
df1=pd.read_csv('traindata1.csv',encoding='utf8')
df2=pd.read_csv('traindata2.csv',encoding='utf8')
trainy1=df1.PPCs
trainx1=df1.drop('PPCs',axis=1)
trainy2=df2.PPCs
trainx2=df2.drop('PPCs',axis=1)

lr= LR(C=2,penalty="l2", 
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
        skip_list = [2,3,4,5,6,7]
        VS=Valuestandard(skip_list, Skip=True)
        VS.fit_transform(patient)
        patient=VS.standard_data_1()
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
            x_train_summary = shap.kmeans(trainx1,2)
            explainer_lr=shap.KernelExplainer(lr.predict_proba,x_train_summary, feature_names=trainx1.columns)
            shap_values= explainer_lr.shap_values(patient)

            _waterfall.waterfall_legacy(explainer_lr.expected_value[0], shap_values[0][1], feature_names=trainx1.columns)
            
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
            skip_list = [4,5,6,7,8,9]
            VS=Valuestandard(skip_list, Skip=True)
            VS.fit_transform(patient)
            patient=VS.standard_data_1()
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
