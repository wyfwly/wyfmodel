import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
import os

#加载保存的模型
model= TabularPredictor.load("AutogluonModels/ag-20260402_113750")
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
"ALT":{"type":"numerical","min":0.000,"max":10000,"default":30},
 "NEUT_Abs":{"type":"numerical","min":0.000,"max":100,"default":6.17},
 "MONO_Pct":{"type":"numerical","min":0.000,"max":1,"default":0.1},
 "BASO_Pct":{"type":"numerical","min":0.000,"max":1,"default":0.01},
 "C1":{"type":"numerical","min":0.000,"max":1000.00,"default":150.0},
 "CREA":{"type":"numerical","min":0.000,"max":1500.00,"default":50.00},
 "C3":{"type":"numerical","min":0.000,"max":60.00,"default":1},
 "SSA_Ab":{"type":"categorical","options":[0,0.5,1],"default":0},
 "Scl70":{"type":"categorical","options":[0,0.5,1],"default":0},
 "Urine_BIL":{"type":"categorical","options":[0,0.5,1],"default":0}
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
    # 1. 准备数据
    required_features = list(feature_ranges.keys())
    background = data[required_features].sample(100, random_state=42)
    input_data = pd.DataFrame([feature_values], columns=required_features)
    
    # 2. 专用预测包装器（处理二分类输出）
    def predict_wrapper(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=required_features)
        proba = model.predict_proba(X)
        # 明确返回正类概率作为一维数组
        return proba.iloc[:, 1].values if hasattr(proba, 'iloc') else proba[:, 1]
    
    # 3. 初始化解释器
    explainer = shap.Explainer(
        predict_wrapper,
        background,
        feature_names=required_features
    )
    
    # 4. 计算SHAP值
    shap_values = explainer(input_data)
    
    # 5. 调试信息（可选）
    st.write("SHAP值形状:", shap_values.shape)
    st.write("SHAP基础值:", shap_values.base_values)
    
    # 6. 生成可视化
    st.subheader("SHAP Force Plot")
    fig = shap.plots.force(
        shap_values[0],  # 直接使用Explanation对象
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
    st.code(f"""
    模型输出类型: {type(model.predict_proba(input_data.head(1)))}
    SHAP值内容: {str(dir(shap_values)) if 'shap_values' in locals() else 'N/A'}
    """, language='python')
