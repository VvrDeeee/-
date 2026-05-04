import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="心血管疾病风险预测系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 侧边栏深蓝配色 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2b4e 0%, #0d3559 50%, #0a2b4e 100%);
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #e8f0f8 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #a8c8e0 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #1a4a7a;
        border-radius: 8px;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #2c6e9e, #1a4a7a);
        border: 1px solid #3a7ca5;
        color: white !important;
    }
    
    /* 右侧主区域浅蓝配色 */
    .main-header {
        background: linear-gradient(135deg, #d0e3f0 0%, #b8d4e8 100%);
    }
    
    /* 保留原有样式，只改颜色 */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffa502, #e67e22);
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #a8e6cf, #3b8d68);
        padding: 15px 20px;
        border-radius: 15px;
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .feature-imp {
        background: #1e1e2f;
        padding: 10px;
        border-radius: 10px;
        color: white;
    }
    .suggestion-box {
        background: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffa502;
        margin: 10px 0;
    }
    
    /* 主区域背景浅蓝 */
    .stApp {
        background: linear-gradient(135deg, #e8f0f8 0%, #d8e8f4 100%);
    }
    
    /* 卡片样式微调 */
    .stTabs [data-baseweb="tab-list"] {
        background: #e8f0f8;
        padding: 0.5rem;
        border-radius: 12px;
        gap: 0.5rem;
    }
    
    /* 按钮悬停效果 */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(26, 74, 122, 0.3);
    }
    
    /* 进度条颜色 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1a4a7a, #3a7ca5);
    }
</style>
""", unsafe_allow_html=True)
# 中文映射
GENDER_CN = {"Female": "女", "Male": "男"}
GENDER_EN = {"女": "Female", "男": "Male"}

HEALTH_CN = {"Excellent": "极好", "Very good": "很好", "Good": "良好", "Fair": "一般", "Poor": "差"}
HEALTH_EN = {v: k for k, v in HEALTH_CN.items()}

SMOKER_CN = {
    "Never smoked": "从不吸烟",
    "Former smoker": "已戒烟",
    "Current smoker - now smokes some days": "偶尔吸烟",
    "Current smoker - now smokes every day": "每天吸烟"
}
SMOKER_EN = {v: k for k, v in SMOKER_CN.items()}

ECIG_CN = {
    "Never used e-cigarettes in my entire life": "从未使用电子烟",
    "Use them some days": "偶尔使用",
    "Use them every day": "每天使用",
    "Not at all (right now)": "目前不使用"
}
ECIG_EN = {v: k for k, v in ECIG_CN.items()}

TEETH_CN = {
    "None of them": "无",
    "1 to 5": "1-5颗",
    "6 or more, but not all": "6颗以上",
    "All": "全部"
}
TEETH_EN = {v: k for k, v in TEETH_CN.items()}

DIABETES_CN = {
    "No": "无",
    "Yes": "有",
    "No, pre-diabetes or borderline diabetes": "糖尿病前期"
}
DIABETES_EN = {v: k for k, v in DIABETES_CN.items()}

TETANUS_CN = {
    "No, did not receive any tetanus shot in the past 10 years": "未接种",
    "Yes, received tetanus shot but not sure what type": "已接种(类型不明)",
    "Yes, received Tdap": "已接种Tdap",
    "Yes, received tetanus shot, but not Tdap": "已接种(非Tdap)"
}
TETANUS_EN = {v: k for k, v in TETANUS_CN.items()}

COVID_CN = {
    "No": "否",
    "Yes": "是",
    "Tested positive using home test without a health professional": "是(自测)"
}
COVID_EN = {v: k for k, v in COVID_CN.items()}

YES_NO_CN = {"No": "否", "Yes": "是"}
YES_NO_EN = {"否": "No", "是": "Yes"}

AGE_CN = {
    "Age 18 to 24": "18-24岁", "Age 25 to 29": "25-29岁", "Age 30 to 34": "30-34岁",
    "Age 35 to 39": "35-39岁", "Age 40 to 44": "40-44岁", "Age 45 to 49": "45-49岁",
    "Age 50 to 54": "50-54岁", "Age 55 to 59": "55-59岁", "Age 60 to 64": "60-64岁",
    "Age 65 to 69": "65-69岁", "Age 70 to 74": "70-74岁", "Age 75 to 79": "75-79岁",
    "Age 80 or older": "80岁及以上"
}
AGE_EN = {v: k for k, v in AGE_CN.items()}

@st.cache_resource
def load_model():
    try:
        with open('xgb_heart_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['encoders'], data['features'], data.get('threshold', 0.5)
    except FileNotFoundError:
        st.error("❌ 未找到模型文件")
        return None, None, None, 0.5

model, encoders, features, threshold = load_model()

# 页面标题
st.title("❤️ 心血管疾病风险预测系统")
st.markdown("基于机器学习算法的心脏健康评估工具")

if model is None:
    st.stop()

# 侧边栏
with st.sidebar:
    st.markdown("## 📖 使用说明")
    st.markdown("""
    1. 填写各项健康信息
    2. 点击「开始评估」按钮
    3. 查看风险评估结果
    4. 参考个性化健康建议
    
    ### ⚠️ 重要提示
    本评估仅供自评参考，不能替代专业医疗诊断。
    """)
    
    st.markdown("---")
    st.markdown("## 🎯 风险等级说明")
    st.markdown("""
    - 🟢 **低风险** (<30%): 保持现状
    - 🟡 **中风险** (30-60%): 建议改善
    - 🔴 **高风险** (>60%): 尽快就医，做系统体检
    """)

# 主界面选项卡
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 基本信息", "🚬 生活习惯", "🩺 疾病史", "👂 身体机能", "💉 其他信息"
])

inputs = {}

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sex_cn = st.selectbox("性别", ["男", "女"], key="sex")
        inputs['Sex'] = GENDER_EN[sex_cn]
        
        age_cn = st.selectbox("年龄段", list(AGE_CN.values()), key="age")
        inputs['AgeCategory'] = AGE_EN[age_cn]
        
        health_cn = st.selectbox("整体健康状况", list(HEALTH_CN.values()), key="health")
        inputs['GeneralHealth'] = HEALTH_EN[health_cn]
    
    with col2:
        height = st.number_input("身高 (米)", min_value=1.0, max_value=2.5, value=1.7, step=0.01, key="height")
        weight = st.number_input("体重 (公斤)", min_value=30.0, max_value=200.0, value=70.0, step=1.0, key="weight")
        bmi = weight / (height ** 2)
        st.metric("体重指数 (BMI)", f"{bmi:.1f}")
        inputs['HeightInMeters'] = height
        inputs['WeightInKilograms'] = weight
        inputs['BMI'] = bmi
    
    with col3:
        physical_days = st.slider("过去30天身体不适天数", 0, 30, 0, key="physical")
        inputs['PhysicalHealthDays'] = physical_days
        
        mental_days = st.slider("过去30天心理不适天数", 0, 30, 0, key="mental")
        inputs['MentalHealthDays'] = mental_days
        
        sleep_hours = st.slider("平均睡眠时长 (小时)", 0, 24, 7, key="sleep")
        inputs['SleepHours'] = sleep_hours

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        smoker_cn = st.selectbox("吸烟状况", list(SMOKER_CN.values()), key="smoker")
        inputs['SmokerStatus'] = SMOKER_EN[smoker_cn]
        
        ecig_cn = st.selectbox("电子烟使用", list(ECIG_CN.values()), key="ecig")
        inputs['ECigaretteUsage'] = ECIG_EN[ecig_cn]
        
        alcohol_cn = st.selectbox("是否饮酒", list(YES_NO_CN.values()), key="alcohol")
        inputs['AlcoholDrinkers'] = YES_NO_EN[alcohol_cn]
    
    with col2:
        teeth_cn = st.selectbox("拔牙数量", list(TEETH_CN.values()), key="teeth")
        inputs['RemovedTeeth'] = TEETH_EN[teeth_cn]
        
        physical_cn = st.selectbox("是否进行体力活动", list(YES_NO_CN.values()), key="physical_act")
        inputs['PhysicalActivities'] = YES_NO_EN[physical_cn]
        
        chest_cn = st.selectbox("是否做过胸部扫描", list(YES_NO_CN.values()), key="chest")
        inputs['ChestScan'] = YES_NO_EN[chest_cn]

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['HadAngina'] = YES_NO_EN[st.selectbox("心绞痛史", list(YES_NO_CN.values()), key="angina")]
        inputs['HadStroke'] = YES_NO_EN[st.selectbox("中风史", list(YES_NO_CN.values()), key="stroke")]
        inputs['HadAsthma'] = YES_NO_EN[st.selectbox("哮喘史", list(YES_NO_CN.values()), key="asthma")]
        inputs['HadSkinCancer'] = YES_NO_EN[st.selectbox("皮肤癌史", list(YES_NO_CN.values()), key="skin")]
    
    with col2:
        inputs['HadCOPD'] = YES_NO_EN[st.selectbox("慢阻肺史", list(YES_NO_CN.values()), key="copd")]
        inputs['HadDepressiveDisorder'] = YES_NO_EN[st.selectbox("抑郁症史", list(YES_NO_CN.values()), key="depression")]
        inputs['HadKidneyDisease'] = YES_NO_EN[st.selectbox("肾病史", list(YES_NO_CN.values()), key="kidney")]
        inputs['HadArthritis'] = YES_NO_EN[st.selectbox("关节炎史", list(YES_NO_CN.values()), key="arthritis")]
        
        diabetes_cn = st.selectbox("糖尿病史", list(DIABETES_CN.values()), key="diabetes")
        inputs['HadDiabetes'] = DIABETES_EN[diabetes_cn]

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        inputs['DeafOrHardOfHearing'] = YES_NO_EN[st.selectbox("听力障碍", list(YES_NO_CN.values()), key="hearing")]
        inputs['BlindOrVisionDifficulty'] = YES_NO_EN[st.selectbox("视力障碍", list(YES_NO_CN.values()), key="vision")]
        inputs['DifficultyConcentrating'] = YES_NO_EN[st.selectbox("难以集中精力", list(YES_NO_CN.values()), key="focus")]
    
    with col2:
        inputs['DifficultyWalking'] = YES_NO_EN[st.selectbox("行走困难", list(YES_NO_CN.values()), key="walking")]
        inputs['DifficultyDressingBathing'] = YES_NO_EN[st.selectbox("自理能力困难", list(YES_NO_CN.values()), key="dressing")]
        inputs['DifficultyErrands'] = YES_NO_EN[st.selectbox("办事行动不便", list(YES_NO_CN.values()), key="errands")]

with tab5:
    col1, col2 = st.columns(2)
    
    with col1:
        hiv_cn = st.selectbox("是否进行过HIV检测", list(YES_NO_CN.values()), key="hiv")
        inputs['HIVTesting'] = YES_NO_EN[hiv_cn]
        
        flu_cn = st.selectbox("过去12个月是否接种流感疫苗", list(YES_NO_CN.values()), key="flu")
        inputs['FluVaxLast12'] = YES_NO_EN[flu_cn]
        
        pneumo_cn = st.selectbox("是否接种过肺炎疫苗", list(YES_NO_CN.values()), key="pneumo")
        inputs['PneumoVaxEver'] = YES_NO_EN[pneumo_cn]
    
    with col2:
        
        
        tetanus_cn = st.selectbox("破伤风疫苗接种情况", list(TETANUS_CN.values()), key="tetanus")
        inputs['TetanusLast10Tdap'] = TETANUS_EN[tetanus_cn]
        
        high_risk_cn = st.selectbox("是否属于高风险人群", list(YES_NO_CN.values()), key="high_risk")
        inputs['HighRiskLastYear'] = YES_NO_EN[high_risk_cn]
        
        covid_cn = st.selectbox("是否曾感染新冠病毒", list(COVID_CN.values()), key="covid")
        inputs['CovidPos'] = COVID_EN[covid_cn]

# 预测按钮
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("开始评估心血管疾病风险", type="primary", use_container_width=True)

# 预测逻辑
if predict_clicked:
    # 输入DataFrame
    input_df = pd.DataFrame([inputs])
    
    # 编码
    for col in features:
        if col in input_df.columns and col in encoders:
            try:
                val = str(input_df[col].iloc[0])
                if val in encoders[col].classes_:
                    input_df[col] = encoders[col].transform([val])[0]
                else:
                    input_df[col] = 0
            except:
                input_df[col] = 0
        elif col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                input_df[col] = 0
        else:
            input_df[col] = 0
    
    # 确保特征顺序正确
    input_df = input_df[[col for col in features if col in input_df.columns]]
    
    # 补全缺失特征
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[features]
    input_df = input_df.fillna(0)
    
    # 预测
    risk_prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if risk_prob >= threshold else 0
    
    risk_percent = risk_prob * 100
    
    # 显示结果
    st.markdown("---")
    st.subheader("📈 评估结果")
    
    if risk_percent >= 60:
        st.markdown(f'<div class="risk-high">', unsafe_allow_html=True)
        st.markdown("### ⚠️ 高风险")
        st.markdown(f"您的风险评估分数为 **{risk_percent:.1f}%**，属于高风险人群。")
        st.markdown("建议尽快咨询专业医生进行全面检查。")
        st.markdown('</div>', unsafe_allow_html=True)
    elif risk_percent >= 30:
        st.markdown(f'<div class="risk-medium">', unsafe_allow_html=True)
        st.markdown("### ⚠️ 中等风险")
        st.markdown(f"您的风险评估分数为 **{risk_percent:.1f}%**，属于中等风险人群。")
        st.markdown("建议改善生活习惯，定期进行健康检查。")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low">', unsafe_allow_html=True)
        st.markdown("### ✅ 低风险")
        st.markdown(f"您的风险评估分数为 **{risk_percent:.1f}%**，属于低风险人群。")
        st.markdown("请继续保持健康的生活方式。")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 风险仪表盘
    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_percent,
            title={"text": "心血管疾病风险指数"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkred" if risk_percent > 60 else "orange" if risk_percent > 30 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "orange"},
                    {'range': [60, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percent
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # 详细指标
    st.markdown("### 📊 详细分析")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1rem; color: #555;">风险评估</div>
            <div style="font-size: 2rem; font-weight: bold; color: #1a4a7a;">{risk_percent:.1f}%</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_m2:
        bmi_status = "正常" if 18.5 <= inputs['BMI'] < 24 else "偏瘦" if inputs['BMI'] < 18.5 else "偏胖" if inputs['BMI'] < 28 else "肥胖"
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1rem; color: #555;">BMI状态</div>
            <div style="font-size: 2rem; font-weight: bold; color: #1a4a7a;">{bmi_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_m3:
        sleep_status = "良好" if 7 <= inputs['SleepHours'] <= 8 else "不足" if inputs['SleepHours'] < 7 else "过多"
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1rem; color: #555;">睡眠质量</div>
            <div style="font-size: 2rem; font-weight: bold; color: #1a4a7a;">{sleep_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_m4:
        activity_status = "活跃" if inputs['PhysicalActivities'] == "Yes" else "缺乏运动"
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 1rem; color: #555;">运动情况</div>
            <div style="font-size: 2rem; font-weight: bold; color: #1a4a7a;">{activity_status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # 个性化建议
    st.markdown("### 💡 个性化健康建议")
    
    suggestions = []
    
    if risk_percent > 60:
        suggestions.append("🔴 **立即行动**：建议尽快预约心脏科医生进行全面检查")
    
    if inputs['BMI'] >= 28:
        suggestions.append("⚖️ **控制体重**：您的BMI偏高，建议通过饮食控制和运动减重")
    elif inputs['BMI'] >= 24:
        suggestions.append("⚖️ **注意体重**：您的BMI偏高，建议适当控制")
    
    if inputs['SmokerStatus'] == "Current smoker - now smokes every day":
        suggestions.append("🚭 **戒烟建议**：吸烟是心血管疾病的主要风险因素，建议尽快戒烟")
    elif inputs['SmokerStatus'] == "Current smoker - now smokes some days":
        suggestions.append("🚭 **减少吸烟**：建议完全戒烟以降低风险")
    
    if inputs['SleepHours'] < 6:
        suggestions.append("😴 **改善睡眠**：睡眠不足会增加心脏负担，建议保证7-8小时睡眠")
    elif inputs['SleepHours'] > 9:
        suggestions.append("😴 **规律作息**：睡眠时间过长也可能与心脏问题相关")
    
    if inputs['PhysicalActivities'] == "No":
        suggestions.append("🏃 **增加运动**：建议每周进行至少150分钟中等强度运动")
    
    if inputs['AlcoholDrinkers'] == "Yes":
        suggestions.append("🍷 **限制饮酒**：过量饮酒会增加心血管疾病风险，建议适量或戒酒")
    
    if inputs['HadDiabetes'] == "Yes":
        suggestions.append("🩸 **控制血糖**：糖尿病是心血管疾病的重要风险因素，请遵医嘱控制血糖")
    
    if inputs['HadDepressiveDisorder'] == "Yes":
        suggestions.append("🧠 **关注心理健康**：抑郁症与心血管疾病风险相关，建议寻求专业帮助")
    
    if not suggestions:
        suggestions.append("✅ **保持现状**：您目前的生活方式良好，请继续保持！")
    
    for s in suggestions:
        st.markdown(f"- {s}")
    
    # 风险因素分析
    st.markdown("### 📋 主要风险因素分析")
    
    risk_factors = []
    if inputs['SmokerStatus'] in ["Current smoker - now smokes every day", "Current smoker - now smokes some days"]:
        risk_factors.append({"因素": "吸烟", "影响": "高风险", "说明": "吸烟会损伤血管内壁，增加血栓风险"})
    if inputs['BMI'] >= 28:
        risk_factors.append({"因素": "肥胖", "影响": "高风险", "说明": "肥胖增加心脏负担，导致高血压、高血脂"})
    elif inputs['BMI'] >= 24:
        risk_factors.append({"因素": "超重", "影响": "中等风险", "说明": "超重会增加心血管疾病风险"})
    if inputs['HadDiabetes'] == "Yes":
        risk_factors.append({"因素": "糖尿病", "影响": "高风险", "说明": "糖尿病会损害血管，增加心血管疾病风险"})
    if inputs['HadAngina'] == "Yes":
        risk_factors.append({"因素": "心绞痛史", "影响": "高风险", "说明": "有心绞痛史说明心脏已存在供血问题"})
    if inputs['HadStroke'] == "Yes":
        risk_factors.append({"因素": "中风史", "影响": "高风险", "说明": "中风与心血管疾病有共同的病理基础"})
    if inputs['PhysicalActivities'] == "No":
        risk_factors.append({"因素": "缺乏运动", "影响": "中等风险", "说明": "缺乏运动会导致心肺功能下降"})
    if inputs['SleepHours'] < 6:
        risk_factors.append({"因素": "睡眠不足", "影响": "中等风险", "说明": "睡眠不足会增加炎症反应和血压"})
    
    if risk_factors:
        risk_df = pd.DataFrame(risk_factors)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    else:
        st.info("未发现明显的主要风险因素，请继续保持健康生活方式！")

# 页脚
st.markdown("---")
st.caption("⚠️ 免责声明：本工具仅供健康参考，不构成医疗建议。如有心脏不适症状，请立即就医。")