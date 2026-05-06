import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="心血管疾病风险预测系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f7ff 0%, #e8f0fa 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2b4e 0%, #0d3559 50%, #0a2b4e 100%);
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #e8f0f8 !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #2c6e9e, #5ba3d0);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .suggestion-item {
        background: #f0f9ff;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #2c6e9e;
    }
    
    .risk-card-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        color: white;
    }
    
    .risk-card-medium {
        background: linear-gradient(135deg, #ffa502, #e67e22);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        color: white;
    }
    
    .risk-card-low {
        background: linear-gradient(135deg, #a8e6cf, #3b8d68);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        color: white;
    }
    
    .risk-factor-high {
        display: inline-block;
        background: #d63031;
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    
    .risk-factor-medium {
        display: inline-block;
        background: #f39c12;
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    
    .risk-factor-low {
        display: inline-block;
        background: #27ae60;
        color: white;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    
    .stButton > button {
        border-radius: 25px;
        font-weight: 500;
    }
    
    hr {
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 中文映射 ====================
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

# ==================== 加载模型 ====================
@st.cache_resource
def load_model():
    try:
        with open('xgb_heart_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['encoders'], data['features'], data.get('threshold', 0.5)
    except FileNotFoundError:
        st.error("❌ 未找到模型文件，请确保 xgb_heart_model.pkl 在正确目录")
        return None, None, None, 0.5

model, encoders, features, threshold = load_model()

# ==================== 初始化session_state ====================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'show_result' not in st.session_state:
    st.session_state.show_result = False

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("## ❤️ 心血管疾病风险预测")
    st.markdown("---")
    
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. 按步骤填写健康信息
    2. 点击「完成评估」按钮
    3. 查看详细的风险评估结果
    
    ---
    
    ### ⚠️ 重要提示
    本评估仅供健康参考，**不能替代**专业医疗诊断。
    
    ---
    
    ### 🎯 风险等级
    - 🟢 **低风险** (<30%)
    - 🟡 **中风险** (30-60%)
    - 🔴 **高风险** (>60%)
    """)
    
    # # 显示进度
    # steps_completed = len([k for k in st.session_state.inputs if k not in ['step', 'show_result']])
    # total_fields = 25
    # progress_pct = min(steps_completed / total_fields, 1.0)
    # st.progress(progress_pct)
    # st.caption(f"已完成 {steps_completed}/{total_fields} 项信息")

# ==================== 主页面标题 ====================
col_title1, col_title2, col_title3 = st.columns([1, 3, 1])
with col_title2:
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="text-align: center; margin: 0;">❤️ 心血管疾病风险预测系统</h1>
        <p style="text-align: center; color: #666;">基于机器学习的心脏健康评估工具</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if model is None:
    st.stop()

# ==================== 反向映射（确保所有字段都有映射） ====================
reverse_maps = {
    "Sex": GENDER_EN,
    "AgeCategory": AGE_EN,
    "GeneralHealth": HEALTH_EN,
    "SmokerStatus": SMOKER_EN,
    "ECigaretteUsage": ECIG_EN,
    "AlcoholDrinkers": YES_NO_EN,
    "PhysicalActivities": YES_NO_EN,
    "ChestScan": YES_NO_EN,
    "HadAngina": YES_NO_EN,
    "HadStroke": YES_NO_EN,
    "HadAsthma": YES_NO_EN,
    "HadSkinCancer": YES_NO_EN,
    "HadCOPD": YES_NO_EN,
    "HadDepressiveDisorder": YES_NO_EN,
    "HadKidneyDisease": YES_NO_EN,
    "HadArthritis": YES_NO_EN,
    "HadDiabetes": DIABETES_EN,
    "RemovedTeeth": TEETH_EN,
    "DeafOrHardOfHearing": YES_NO_EN,
    "BlindOrVisionDifficulty": YES_NO_EN,
    "DifficultyConcentrating": YES_NO_EN,
    "DifficultyWalking": YES_NO_EN,
    "DifficultyDressingBathing": YES_NO_EN,
    "DifficultyErrands": YES_NO_EN,
    "HIVTesting": YES_NO_EN,
    "FluVaxLast12": YES_NO_EN,
    "PneumoVaxEver": YES_NO_EN,
    "TetanusLast10Tdap": TETANUS_EN,
    "HighRiskLastYear": YES_NO_EN,
    "CovidPos": COVID_EN
}

# ==================== 步骤定义（完整4步，共25个字段） ====================
step_fields = {
    1: {
        "title": "📊 基本信息",
        "icon": "👤",
        "description": "请填写您的基本信息和身体状况",
        "fields": [
            ("Sex", "性别", "select", ["男", "女"]),
            ("AgeCategory", "年龄段", "select", list(AGE_CN.values())),
            ("GeneralHealth", "整体健康状况", "select", list(HEALTH_CN.values())),
            ("HeightInMeters", "身高(米)", "number", (1.0, 2.5, 1.7)),
            ("WeightInKilograms", "体重(公斤)", "number", (30.0, 200.0, 70.0)),
            ("PhysicalHealthDays", "过去30天身体不适天数", "slider", (0, 30, 0)),
            ("MentalHealthDays", "过去30天心理不适天数", "slider", (0, 30, 0)),
            ("SleepHours", "平均睡眠时长(小时)", "slider", (0, 24, 7))
        ]
    },
    2: {
        "title": "🚬 生活习惯",
        "icon": "🏃",
        "description": "这些信息有助于评估生活方式对心脏健康的影响",
        "fields": [
            ("SmokerStatus", "吸烟状况", "select", list(SMOKER_CN.values())),
            ("ECigaretteUsage", "电子烟使用", "select", list(ECIG_CN.values())),
            ("AlcoholDrinkers", "是否饮酒", "select", list(YES_NO_CN.values())),
            ("RemovedTeeth", "拔牙数量", "select", list(TEETH_CN.values())),
            ("PhysicalActivities", "是否进行体力活动", "select", list(YES_NO_CN.values())),
            ("ChestScan", "是否做过胸部扫描", "select", list(YES_NO_CN.values()))
        ]
    },
    3: {
        "title": "🩺 疾病史",
        "icon": "🏥",
        "description": "既往病史是评估心血管风险的重要因素",
        "fields": [
            ("HadAngina", "心绞痛史", "select", list(YES_NO_CN.values())),
            ("HadStroke", "中风史", "select", list(YES_NO_CN.values())),
            ("HadAsthma", "哮喘史", "select", list(YES_NO_CN.values())),
            ("HadSkinCancer", "皮肤癌史", "select", list(YES_NO_CN.values())),
            ("HadCOPD", "慢阻肺史", "select", list(YES_NO_CN.values())),
            ("HadDepressiveDisorder", "抑郁症史", "select", list(YES_NO_CN.values())),
            ("HadKidneyDisease", "肾病史", "select", list(YES_NO_CN.values())),
            ("HadArthritis", "关节炎史", "select", list(YES_NO_CN.values())),
            ("HadDiabetes", "糖尿病史", "select", list(DIABETES_CN.values()))
        ]
    },
    4: {
        "title": "👂 身体机能与其他",
        "icon": "💪",
        "description": "这些信息帮助全面评估健康状况",
        "fields": [
            ("DeafOrHardOfHearing", "听力障碍", "select", list(YES_NO_CN.values())),
            ("BlindOrVisionDifficulty", "视力障碍", "select", list(YES_NO_CN.values())),
            ("DifficultyConcentrating", "难以集中精力", "select", list(YES_NO_CN.values())),
            ("DifficultyWalking", "行走困难", "select", list(YES_NO_CN.values())),
            ("DifficultyDressingBathing", "自理能力困难", "select", list(YES_NO_CN.values())),
            ("DifficultyErrands", "办事行动不便", "select", list(YES_NO_CN.values())),
            ("HIVTesting", "HIV检测史", "select", list(YES_NO_CN.values())),
            ("FluVaxLast12", "流感疫苗接种(过去12个月)", "select", list(YES_NO_CN.values())),
            ("PneumoVaxEver", "肺炎疫苗接种史", "select", list(YES_NO_CN.values())),
            ("TetanusLast10Tdap", "破伤风疫苗接种情况", "select", list(TETANUS_CN.values())),
            ("HighRiskLastYear", "是否属于高风险人群", "select", list(YES_NO_CN.values())),
            ("CovidPos", "是否曾感染新冠病毒", "select", list(COVID_CN.values()))
        ]
    }
}

# ==================== 风险因素等级判断（只分类，不给分值） ====================
def get_risk_level(inputs):
    """判断各风险因素的等级（高/中/低），不给出具体数值"""
    risk_factors = []
    
    # 吸烟
    smoker = inputs.get('SmokerStatus', '')
    if smoker in ["Current smoker - now smokes every day", "Current smoker - now smokes some days"]:
        risk_factors.append({"因素": "吸烟", "等级": "高", "说明": "吸烟会损伤血管内壁，增加血栓风险"})
    elif smoker == "Former smoker":
        risk_factors.append({"因素": "吸烟史", "等级": "中", "说明": "既往吸烟史仍有影响，但戒烟会逐步降低风险"})
    
    # BMI
    bmi = inputs.get('BMI', 22)
    if bmi >= 28:
        risk_factors.append({"因素": "肥胖", "等级": "高", "说明": "肥胖增加心脏负担，可能导致高血压、高血脂"})
    elif bmi >= 24:
        risk_factors.append({"因素": "超重", "等级": "中", "说明": "超重会增加心血管疾病风险"})
    
    # 糖尿病
    diabetes = inputs.get('HadDiabetes', '')
    if diabetes == "Yes":
        risk_factors.append({"因素": "糖尿病", "等级": "高", "说明": "糖尿病会损害血管，增加心血管疾病风险"})
    elif diabetes == "No, pre-diabetes or borderline diabetes":
        risk_factors.append({"因素": "糖尿病前期", "等级": "中", "说明": "糖尿病前期需要警惕，建议控制饮食"})
    
    # 心绞痛史
    if inputs.get('HadAngina') == "Yes":
        risk_factors.append({"因素": "心绞痛史", "等级": "高", "说明": "有心绞痛史说明心脏已存在供血问题"})
    
    # 中风史
    if inputs.get('HadStroke') == "Yes":
        risk_factors.append({"因素": "中风史", "等级": "高", "说明": "中风与心血管疾病有共同的病理基础"})
    
    # 缺乏运动
    if inputs.get('PhysicalActivities') == "No":
        risk_factors.append({"因素": "缺乏运动", "等级": "中", "说明": "缺乏运动会导致心肺功能下降"})
    
    # 睡眠问题
    sleep_hours = inputs.get('SleepHours', 7)
    if sleep_hours < 6:
        risk_factors.append({"因素": "睡眠不足", "等级": "中", "说明": "睡眠不足会增加炎症反应和血压"})
    elif sleep_hours > 9:
        risk_factors.append({"因素": "睡眠过长", "等级": "低", "说明": "睡眠时间过长也可能与心脏问题相关"})
    
    # 饮酒
    if inputs.get('AlcoholDrinkers') == "Yes":
        risk_factors.append({"因素": "饮酒", "等级": "中", "说明": "过量饮酒会增加心血管疾病风险"})
    
    # 抑郁症
    if inputs.get('HadDepressiveDisorder') == "Yes":
        risk_factors.append({"因素": "抑郁症", "等级": "中", "说明": "抑郁症与心血管疾病风险相关"})
    
    # 肾病
    if inputs.get('HadKidneyDisease') == "Yes":
        risk_factors.append({"因素": "慢性肾病", "等级": "高", "说明": "肾病与心血管疾病密切相关"})
    
    return risk_factors

# ==================== 表单处理 ====================
if not st.session_state.show_result:
    step_info = step_fields[st.session_state.step]
    
    # 步骤标题卡片
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #e8f0f8, #d4e4f0); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h2 style="margin:0;">{step_info['icon']} {step_info['title']}</h2>
        <p style="margin:5px 0 0 0; color:#555;">📌 {step_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示进度
    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.markdown(f"<p style='text-align:center; font-size:0.9rem;'>第 {st.session_state.step}/4 步</p>", unsafe_allow_html=True)
        st.progress(st.session_state.step / 4)
    
    # 显示当前步骤的表单字段
    with st.form(key=f"step_{st.session_state.step}"):
        field_values = {}
        
        # 使用两列布局
        form_cols = st.columns(2)
        
        for idx, (field_key, field_label, field_type, field_options) in enumerate(step_info["fields"]):
            with form_cols[idx % 2]:
                if field_type == "select":
                    # 获取当前已保存的值
                    current_val = st.session_state.inputs.get(field_key)
                    if current_val:
                        # 将英文值转回中文显示
                        reverse_map = reverse_maps.get(field_key, {})
                        for cn_val, en_val in reverse_map.items():
                            if en_val == current_val:
                                current_val = cn_val
                                break
                    
                    selected = st.selectbox(
                        field_label,
                        field_options,
                        index=field_options.index(current_val) if current_val in field_options else 0,
                        key=f"step{st.session_state.step}_{field_key}"
                    )
                    field_values[field_key] = selected
                
                elif field_type == "number":
                    min_val, max_val, default_val = field_options
                    value = st.number_input(
                        field_label,
                        min_value=min_val,
                        max_value=max_val,
                        value=st.session_state.inputs.get(field_key, default_val),
                        step=0.01 if "米" in field_label else 1.0,
                        key=f"step{st.session_state.step}_{field_key}"
                    )
                    field_values[field_key] = value
                
                elif field_type == "slider":
                    min_val, max_val, default_val = field_options
                    value = st.slider(
                        field_label,
                        min_val, max_val,
                        st.session_state.inputs.get(field_key, default_val),
                        key=f"step{st.session_state.step}_{field_key}"
                    )
                    field_values[field_key] = value
        
        # 计算BMI
        if st.session_state.step == 1:
            height = field_values.get("HeightInMeters", 1.7)
            weight = field_values.get("WeightInKilograms", 70.0)
            bmi = weight / (height ** 2)
            # st.info(f"📊 您的BMI指数：**{bmi:.1f}** - " +   
            #        ("体重偏轻" if bmi < 18.5 else "正常范围" if bmi < 24 else "超重" if bmi < 28 else "肥胖"))
            field_values["BMI"] = bmi
        
        # 表单按钮
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.session_state.step > 1:
                if st.form_submit_button("← 上一步", use_container_width=True):
                    st.session_state.step -= 1
                    st.rerun()
        
        with col2:
            if st.form_submit_button("保存并继续 →", type="primary", use_container_width=True):
                # 保存当前步骤的值
                for key, value in field_values.items():
                    if key == "BMI":
                        st.session_state.inputs[key] = value
                    else:
                        # 转换中文值为英文
                        reverse_map = reverse_maps.get(key, {})
                        # 直接判断 value 是否在映射中
                        if value in reverse_map:
                            st.session_state.inputs[key] = reverse_map[value]
                        else:
                            st.session_state.inputs[key] = value
                
                if st.session_state.step < 4:
                    st.session_state.step += 1
                    st.rerun()
                else:
                    st.session_state.show_result = True
                    st.rerun()
        
        with col3:
            if st.form_submit_button("重新开始", use_container_width=True):
                st.session_state.inputs = {}
                st.session_state.step = 1
                st.session_state.show_result = False
                st.rerun()

# ==================== 结果显示 ====================
if st.session_state.show_result:
    inputs = st.session_state.inputs
    
    # 检查是否有足够的输入
    if len(inputs) < 20:
        st.warning("⚠️ 数据不完整，请重新填写")
        if st.button("重新开始评估"):
            st.session_state.inputs = {}
            st.session_state.step = 1
            st.session_state.show_result = False
            st.rerun()
        st.stop()
    
    
    # ========== 预测逻辑（完全复制原始代码，确保结果一致） ==========
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
    risk_percent = risk_prob * 100
    
    # 获取显示用数据
    bmi = inputs.get('BMI', 24)
    bmi_status = "正常" if 18.5 <= bmi < 24 else "偏瘦" if bmi < 18.5 else "超重" if bmi < 28 else "肥胖"
    sleep_status = "良好" if 7 <= inputs.get('SleepHours', 7) <= 8 else "不足" if inputs.get('SleepHours', 7) < 7 else "过多"
    activity_status = "活跃" if inputs.get('PhysicalActivities') == "Yes" else "缺乏运动"
    
    # 获取吸烟状况的中文
    smoker_val = inputs.get('SmokerStatus', '')
    smoker_cn = ""
    for cn, en in SMOKER_CN.items():
        if en == smoker_val:
            smoker_cn = cn
            break
    
    # ==================== 结果显示区域 ====================
    st.markdown("## 📈 评估结果")
    
    # 风险卡片
    if risk_percent >= 60:
        st.markdown(f"""
        <div class="risk-card-high">
            <div style="font-size: 4rem;">🔴⚠️</div>
            <div style="font-size: 2rem; font-weight: bold;">高风险</div>
            <div style="font-size: 3rem; font-weight: bold; margin: 15px 0;">{risk_percent:.1f}%</div>
            <div style="font-size: 1.1rem;">您的风险评估分数较高，属于高风险人群</div>
            <div style="margin-top: 15px;">⚠️ 建议尽快咨询专业医生进行全面检查</div>
        </div>
        """, unsafe_allow_html=True)
    elif risk_percent >= 30:
        st.markdown(f"""
        <div class="risk-card-medium">
            <div style="font-size: 4rem;">🟡⚠️</div>
            <div style="font-size: 2rem; font-weight: bold;">中等风险</div>
            <div style="font-size: 3rem; font-weight: bold; margin: 15px 0;">{risk_percent:.1f}%</div>
            <div style="font-size: 1.1rem;">您的风险评估分数处于中等水平</div>
            <div style="margin-top: 15px;">💡 建议改善生活习惯，定期进行健康检查</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-card-low">
            <div style="font-size: 4rem;">🟢✅</div>
            <div style="font-size: 2rem; font-weight: bold;">低风险</div>
            <div style="font-size: 3rem; font-weight: bold; margin: 15px 0;">{risk_percent:.1f}%</div>
            <div style="font-size: 1.1rem;">您的风险评估分数较低，属于低风险人群</div>
            <div style="margin-top: 15px;">🌟 请继续保持健康的生活方式</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 风险仪表盘
    st.markdown("### 🎯 风险可视化")
    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            title={"text": "心血管疾病风险指数"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkred" if risk_percent > 60 else "orange" if risk_percent > 30 else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 60], 'color': "#fff3cd"},
                    {'range': [60, 100], 'color': "#f8d7da"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # 详细指标卡片
    st.markdown("### 📊 健康指标概览")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #2c6e9e;">风险评分</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #1a4a7a;">{risk_percent:.1f}%</div>
            <div style="font-size: 0.8rem; color: {'#d63031' if risk_percent>=60 else '#e67e22' if risk_percent>=30 else '#00b894'};">
                {'🔴 高风险' if risk_percent>=60 else '🟡 中风险' if risk_percent>=30 else '🟢 低风险'}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #2c6e9e;">BMI指数</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #1a4a7a;">{bmi:.1f}</div>
            <div style="font-size: 0.8rem; color: #555;">{bmi_status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #2c6e9e;">睡眠质量</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #1a4a7a;">{inputs.get('SleepHours', 7)}h</div>
            <div style="font-size: 0.8rem; color: #555;">{sleep_status}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #2c6e9e;">运动情况</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #1a4a7a;">{activity_status}</div>
            <div style="font-size: 0.8rem; color: #555;">{'✅ 每周运动' if activity_status=='活跃' else '⚠️ 建议增加运动'}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # 风险因素分析（只显示高中低分类，不给分值）
    st.markdown("### 📋 风险因素分析")
    
    risk_factors = get_risk_level(inputs)
    
    if risk_factors:
        for rf in risk_factors:
            level = rf["等级"]
            if level == "高":
                badge = '<span class="risk-factor-high">高风险</span>'
            elif level == "中":
                badge = '<span class="risk-factor-medium">中风险</span>'
            else:
                badge = '<span class="risk-factor-low">低风险</span>'
            
            st.markdown(f"""
            <div style="padding: 10px 0; border-bottom: 1px solid #eee;">
                <span style="font-weight: bold;">{rf['因素']}</span> {badge}<br>
                <span style="font-size: 0.85rem; color: #666;">{rf['说明']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # 添加总结
        high_count = len([r for r in risk_factors if r["等级"] == "高"])
        medium_count = len([r for r in risk_factors if r["等级"] == "中"])
        
        if high_count > 0:
            st.warning(f"⚠️ 您有 {high_count} 项高风险因素，建议尽快咨询医生")
        elif medium_count > 2:
            st.info(f"📌 您有 {medium_count} 项中风险因素，建议改善生活习惯")
        elif medium_count > 0:
            st.info(f"📌 您有 {medium_count} 项中风险因素，注意日常保健")
    else:
        st.info("✅ 未发现明显的风险因素，请继续保持健康生活方式！")
    
    # 个性化建议
    st.markdown("### 💡 个性化健康建议")
    
    suggestions = []
    
    if risk_percent > 60:
        suggestions.append("🔴 **立即行动**：建议尽快预约心脏科医生进行全面检查")
    
    if bmi >= 28:
        suggestions.append("⚖️ **控制体重**：您的BMI偏高，建议通过饮食控制和运动减重")
    elif bmi >= 24:
        suggestions.append("⚖️ **注意体重**：您的BMI偏高，建议适当控制")
    
    if smoker_val in ["Current smoker - now smokes every day", "Current smoker - now smokes some days"]:
        suggestions.append("🚭 **戒烟建议**：吸烟是心血管疾病的主要风险因素，建议尽快戒烟")
    elif smoker_val == "Former smoker":
        suggestions.append("🚭 **保持戒烟**：您已戒烟，这对心脏健康非常有益")
    
    if inputs.get('SleepHours', 7) < 6:
        suggestions.append("😴 **改善睡眠**：睡眠不足会增加心脏负担，建议保证7-8小时睡眠")
    elif inputs.get('SleepHours', 7) > 9:
        suggestions.append("😴 **规律作息**：睡眠时间过长也可能与心脏问题相关")
    
    if inputs.get('PhysicalActivities') == "No":
        suggestions.append("🏃 **增加运动**：建议每周进行至少150分钟中等强度运动")
    
    if inputs.get('AlcoholDrinkers') == "Yes":
        suggestions.append("🍷 **限制饮酒**：过量饮酒会增加心血管疾病风险，建议适量或戒酒")
    
    if inputs.get('HadDiabetes') == "Yes":
        suggestions.append("🩸 **控制血糖**：糖尿病是心血管疾病的重要风险因素，请遵医嘱控制血糖")
    
    if inputs.get('HadDepressiveDisorder') == "Yes":
        suggestions.append("🧠 **关注心理健康**：抑郁症与心血管疾病风险相关，建议寻求专业帮助")
    
    if inputs.get('HadKidneyDisease') == "Yes":
        suggestions.append("🩺 **肾功能管理**：慢性肾病与心血管疾病密切相关，请定期随访")
    
    if inputs.get('HadAngina') == "Yes":
        suggestions.append("❤️ **心脏专科随访**：有心绞痛史的患者，建议定期心脏科复查")
    
    if inputs.get('PhysicalHealthDays', 0) > 10:
        suggestions.append("💊 **关注身体不适**：过去30天内身体不适天数较多，建议咨询医生")
    
    if inputs.get('MentalHealthDays', 0) > 10:
        suggestions.append("🧘 **关注心理状态**：心理健康同样重要，建议适当放松和减压")
    
    if not suggestions:
        suggestions.append("✅ **保持现状**：您目前的生活方式良好，请继续保持！")
    
    for s in suggestions:
        st.markdown(f'<div class="suggestion-item">{s}</div>', unsafe_allow_html=True)
    
    # 重新评估按钮
    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        if st.button("🔄 重新开始评估", use_container_width=True):
            st.session_state.inputs = {}
            st.session_state.step = 1
            st.session_state.show_result = False
            st.rerun()

# ==================== 页脚 ====================
st.markdown("---")
st.caption("⚠️ 免责声明：本工具仅供健康参考，不构成医疗建议。如有心脏不适症状，请立即就医。")
st.caption("💡 本系统基于机器学习算法，评估结果仅供参考")
