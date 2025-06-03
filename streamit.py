import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
import os

#加载保存的模型
model= TabularPredictor.load("AutogluonModels/ag-20250426_142741")
data = pd.read_csv("finaldata.csv")

class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names

    def predict_proba(self, X):
        """将输入转换为AutoGluon需要的格式并返回概率预测"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X).values
#特征范围定义（根据提供的特征范围和数据类型）
feature_ranges ={
 "BASO,10^9/L":{"type":"numerical","min":0.000,"max":0.1,"default":0.02},
 "CHE,U/L":{"type":"numerical","min":0.000,"max":15.00,"default":6.17},
 "MPV,fL":{"type":"numerical","min":0.000,"max":20.00,"default":10.73},
 "Hb,g/L":{"type":"numerical","min":0.000,"max":200,"default":110.0},
 "PLT,10^9/L":{"type":"numerical","min":0.000,"max":1000.00,"default":150.0},
 "u.WBC,/uL":{"type":"numerical","min":0.000,"max":1500.00,"default":50.00},
 "GLU,mmol/L":{"type":"numerical","min":0.000,"max":60.00,"default":6.00},
 "MCV,fL":{"type":"numerical","min":0.000,"max":150.00,"default":90.00},
 "GGT,U/L":{"type":"numerical","min":0.000,"max":1000.00,"default":50.00},
 "RDW-SD,fL":{"type":"numerical","min":0.000,"max":200,"default":50.00},
 "LY%":{"type":"numerical","min":0.000,"max":100.00,"default":24.00},
 "age,year":{"type":"numerical","min":18.00,"max":100.00,"default":43.00},
 "aRO52(1:100)":{"type":"categorical","options":[0,1,2,3],"default":0}
}
#Streamlit 界面
st.title (" Prediction Model with SHAP Visualization ")
#动态生成输入项
st.header ("Enter the following feature values :")
feature_values = []
for feature, properties in feature_ranges.items():
    # 初始化value为None或合适的默认值
    value = None

    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
            index=properties["options"].index(properties["default"]),
        )

    # 确保value已被赋值
    if value is not None:
        feature_values.append(value)
    else:
        st.warning(f"未处理的特征类型: {feature}")

features = np.array([feature_values])
 # 预测与 SHAP 可视化
if st.button("Predict"):
    # 将输入数据转为DataFrame（AutoGluon需要）
    input_data = pd.DataFrame([feature_values], columns=feature_ranges.keys())

    # 获取预测概率
    wrapper = AutogluonWrapper(model, list(feature_ranges.keys()))
    proba_df = wrapper.predict_proba(input_data)
    probability = proba_df[0][1] * 100  # 获取正类的概率

    # 显示预测结果
    text = f"Based on feature values, predicted possibility of CVD is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    st.pyplot(fig)

# 计算SHAP值
try:
    # 1. 准备背景数据
    background = data[list(feature_ranges.keys())].sample(100, random_state=42)
    
    # 2. 创建预测函数包装器
    def predict_proba_wrapper(X):
        if isinstance(X, pd.DataFrame):
            return model.predict_proba(X)
        return model.predict_proba(pd.DataFrame(X, columns=feature_ranges.keys()))
    
    # 3. 初始化解释器
    explainer = shap.KernelExplainer(
        predict_proba_wrapper,
        background
    )
    
    # 4. 计算SHAP值
    shap_values = explainer.shap_values(input_data)
    expected_value = explainer.expected_value
    
    # 5. 统一SHAP值格式处理
    if isinstance(shap_values, list):
        # 如果是列表，取第一个元素（适用于二分类）
        shap_values = shap_values[0]
        expected_value = expected_value[0]
    elif len(shap_values.shape) == 3:
        # 如果是三维数组（多分类），取第一个类别
        shap_values = shap_values[0]
        expected_value = expected_value[0]
    
    # 6. 确保是一维数组
    if len(shap_values.shape) == 2:
        shap_values = shap_values[0]  # 取第一个样本
    
    # 7. 生成SHAP力图
    st.subheader("SHAP Force Plot")
    fig = shap.force_plot(
        expected_value,
        shap_values,
        input_data.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    
    # 8. 生成SHAP摘要图
    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values.reshape(1, -1),  # 确保是二维
        background,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error generating SHAP explanation: {str(e)}")
