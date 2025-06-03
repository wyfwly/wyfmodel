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
    # 1. 确保特征一致性
    required_features = list(feature_ranges.keys())
    background = data[required_features].sample(100, random_state=42)
    input_data = pd.DataFrame([feature_values], columns=required_features)
    
    # 2. 更安全的预测包装器
    def predict_wrapper(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=required_features)
        proba = model.predict_proba(X)
        # 处理二分类和多分类情况
        return proba.values if hasattr(proba, 'values') else proba
    
    # 3. 初始化解释器（使用最新API）
    explainer = shap.Explainer(
        predict_wrapper,
        background,
        feature_names=required_features
    )
    
    # 4. 计算SHAP值（显式处理输出）
    shap_values = explainer(input_data)
    
    # 5. 统一SHAP值格式
    if len(shap_values.shape) == 3:  # 多分类输出 (n_classes, n_samples, n_features)
        shap_values = shap_values[1]  # 取正类解释
    elif len(shap_values.shape) == 2:  # 二分类输出 (n_samples, n_features)
        pass  # 直接使用
    
    # 6. 生成可视化
    st.subheader("SHAP Force Plot")
    fig = shap.plots.force(
        shap_values[0],  # 第一个样本
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
    
    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(
        shap_values,
        show=False
    )
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error generating SHAP explanation: {str(e)}")
    # 详细调试信息
    st.code(f"""
    错误详情: {repr(e)}
    输入数据形状: {input_data.shape if 'input_data' in locals() else 'N/A'}
    模型输出示例: {model.predict_proba(input_data.head(1)) if 'input_data' in locals() else 'N/A'}
    SHAP值类型: {type(shap_values) if 'shap_values' in locals() else 'N/A'}
    """, language='python')
