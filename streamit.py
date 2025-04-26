import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

#加载保存的随机森林模型
model = TabularPredictor.load("predictor.pkl") 





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
# 动态输入
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']}-{properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=feature,
            options=properties["options"],
            index=properties["options"].index(properties["default"]),
        )
    feature_values.append(value)
 #预测与 SHAP 可视化
# 预测
if st.button("Predict"):
    input_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 获取概率预测
    proba_df = model.predict_proba(input_df)
    probability = proba_df.iloc[0, 1]*100  # 取正例概率 
#显示预测结果，使用 Matplotlib 渲染指字体
text = f" Based on feature values , predicted possibility of CVD is ( probability:.2f)%"
fig, ax = plt.subplots (figsize =(8,1))
ax.text (
0.5,0.5,text,
 fontsize =16,
 ha ='center', va ='center',
 fontname ='Times New Roman',
 transform = ax.transAxes
 )
ax.axis ('off')
plt.savefig (" prediction_text.png", bbox_inches ='tight', dpi =300)
st.image ("prediction_text.png ")
#计算 SHAP
explainer = shap.KernelExplainer(model)
shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns = feature_ranges.keys ()))

#生成 SHAP 力图
class_index = predicted_class #当前预测类别
shap_fig = shap.force_plot (
 explainer.expected_value[class_index],
 shap_values[:,:,class_index],
 pd.DataFrame([feature_values], columns = feature_ranges.keys ()),
 matplotlib = True,
)
#保存并显示 SHAP 图
plt.savefig (" shap_force_plot.png",bbox_inches =' tight ', dpi =1200)
st.image (" shap_force_plot.png ")
