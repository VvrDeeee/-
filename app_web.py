import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="心血管疾病风险预测系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&display=swap');

    /* ==================== 全局星空背景 ==================== */
    .stApp {
        background: linear-gradient(160deg, #020111 0%, #050520 15%, #0a0e27 30%, #0d1b3e 50%, #0f2557 70%, #09101e 100%) !important;
        color: #e0e8ff !important;
        overflow-x: hidden;
    }

    /* ==================== 顶部 Header 栏 ==================== */
    header[data-testid="stHeader"] {
        background: rgba(4,12,30,0.95) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(0,150,255,0.15);
        box-shadow: 0 2px 20px rgba(0,80,180,0.12);
    }
    header[data-testid="stHeader"] button {
        color: #90b8d8 !important;
    }
    header[data-testid="stHeader"] button:hover {
        color: #c0ddf8 !important;
        background: rgba(0,130,220,0.12) !important;
    }

    /* 星空 Canvas 画布 */
    #star-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        pointer-events: none;
    }

    /* 所有主要内容置于星空之上 */
    .main .block-container {
        position: relative;
        z-index: 1;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* ==================== 科技感线条装饰 ==================== */
    .tech-line-top, .tech-line-bottom {
        position: fixed;
        left: 0;
        width: 100%;
        height: 2px;
        z-index: 2;
        pointer-events: none;
    }
    .tech-line-top {
        top: 0;
        background: linear-gradient(90deg,
            transparent 0%, rgba(0,180,255,0.1) 5%, rgba(0,180,255,0.4) 20%,
            rgba(0,220,255,0.5) 50%, rgba(0,180,255,0.4) 80%, rgba(0,180,255,0.1) 95%, transparent 100%);
        box-shadow: 0 0 15px rgba(0,180,255,0.3), 0 0 30px rgba(0,150,255,0.15);
    }
    .tech-line-bottom {
        bottom: 0;
        background: linear-gradient(90deg,
            transparent 0%, rgba(0,180,255,0.05) 10%, rgba(0,220,255,0.3) 50%, rgba(0,180,255,0.05) 90%, transparent 100%);
        box-shadow: 0 0 8px rgba(0,180,255,0.2);
    }

    /* ==================== 左侧导航栏 - 深蓝科技风 ==================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #040a1a 0%, #080e2e 30%, #0a1233 60%, #060d25 100%) !important;
        border-right: 1px solid rgba(0,180,255,0.15) !important;
        box-shadow: 4px 0 30px rgba(0,100,200,0.1) !important;
        position: relative;
        z-index: 10;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 1px;
        height: 100%;
        background: linear-gradient(180deg,
            transparent 0%, rgba(0,200,255,0.3) 30%, rgba(0,200,255,0.6) 50%, rgba(0,200,255,0.3) 70%, transparent 100%);
        pointer-events: none;
    }
    [data-testid="stSidebar"] * {
        color: #c8d8f0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #60c0ff !important;
        text-shadow: 0 0 10px rgba(96,192,255,0.4);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(0,150,255,0.2) !important;
        box-shadow: 0 0 8px rgba(0,150,255,0.1);
    }
    [data-testid="stSidebar"] .st-emotion-cache-1v7f65g .e1b2p2ww15 {
        border-color: rgba(0,150,255,0.15);
    }

    /* ==================== 标题区 ==================== */
    .main-title-glow {
        font-family: 'Orbitron', 'Microsoft YaHei', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60c0ff 0%, #a0e0ff 40%, #ffffff 55%, #80d0ff 70%, #40a0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: none;
        filter: drop-shadow(0 0 20px rgba(100,180,255,0.5));
        letter-spacing: 2px;
    }
    .subtitle-glow {
        color: #80b0d0;
        letter-spacing: 3px;
        font-size: 0.9rem;
        opacity: 0.85;
        text-shadow: 0 0 8px rgba(100,160,220,0.3);
    }

    /* ==================== 步骤卡片 - 玻璃拟态 ==================== */
    .step-header-card {
        background: linear-gradient(135deg,
            rgba(10,25,60,0.75) 0%, rgba(15,35,80,0.65) 50%, rgba(10,25,60,0.75) 100%) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px 28px;
        border-radius: 16px;
        margin-bottom: 30px;
        border: 1px solid rgba(0,180,255,0.2);
        box-shadow: 0 4px 30px rgba(0,100,200,0.15), inset 0 1px 0 rgba(255,255,255,0.03);
        position: relative;
        overflow: hidden;
    }
    .step-header-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 20%, rgba(0,180,255,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .step-header-card h2 {
        color: #60c0ff !important;
        text-shadow: 0 0 12px rgba(96,192,255,0.3);
    }
    .step-header-card p {
        color: #8090b0 !important;
    }

    /* ==================== 通用卡片 ==================== */
    .glass-card {
        background: linear-gradient(135deg,
            rgba(10,25,55,0.7) 0%, rgba(15,35,75,0.6) 100%) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0,160,255,0.18);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 25px rgba(0,80,180,0.12);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0,180,255,0.35);
        box-shadow: 0 6px 35px rgba(0,100,200,0.2);
    }

    /* ==================== 表单输入框美化 ==================== */
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: rgba(8,20,50,0.7) !important;
        border: 1.5px solid rgba(80,190,255,0.45) !important;
        border-radius: 10px !important;
        color: #d0e0ff !important;
        backdrop-filter: blur(8px);
    }
    .stSelectbox > div > div:focus-within, .stNumberInput > div > div > input:focus {
        border-color: rgba(80,210,255,0.75) !important;
        box-shadow: 0 0 18px rgba(0,180,255,0.25), 0 0 40px rgba(0,180,255,0.08) !important;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #90b0d0 !important;
        font-weight: 500;
    }

    /* 数字输入框 - 暴力覆盖所有内层元素 */
    .stNumberInput input[type="number"] {
        background: rgba(8,20,50,0.85) !important;
        border: 1.5px solid rgba(80,190,255,0.45) !important;
        border-radius: 8px !important;
        color: #d0e0ff !important;
    }
    .stNumberInput input[type="number"]:focus {
        border-color: rgba(80,210,255,0.75) !important;
        box-shadow: 0 0 18px rgba(0,180,255,0.25), 0 0 40px rgba(0,180,255,0.08) !important;
    }
    [data-testid="stNumberInputContainer"] {
        background: transparent !important;
    }
    [data-testid="stNumberInputContainer"] button {
        background: rgba(12,30,60,0.9) !important;
        border: 1px solid rgba(0,150,255,0.25) !important;
        color: #a0c8f0 !important;
        border-radius: 6px !important;
    }
    [data-testid="stNumberInputContainer"] button:hover {
        background: rgba(20,45,85,0.9) !important;
        border-color: rgba(0,200,255,0.5) !important;
        color: #d0e8ff !important;
    }
    [data-testid="stNumberInputContainer"] button svg {
        fill: #a0c8f0 !important;
    }
    /* 覆盖数字输入框内所有中间层 div */
    [data-testid="stNumberInputContainer"] > div,
    [data-testid="stNumberInputContainer"] > div > div,
    [data-testid="stNumberInputContainer"] > div > div > div {
        background: transparent !important;
    }

    /* 滑块美化 */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #1a3a6a, #2080c0) !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        background: linear-gradient(135deg, #1a4a8a, #3090d0) !important;
        color: #fff !important;
        border-radius: 6px !important;
    }

    /* ==================== 按钮美化 ==================== */
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        border-radius: 25px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
    }
    /* 主按钮 (包含 form submit primary) */
    button[kind="primary"], button[kind="primaryFormSubmit"],
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a5a9a 0%, #2080d0 50%, #1a5a9a 100%) !important;
        border: 1px solid rgba(80,180,255,0.4) !important;
        color: #e0f0ff !important;
        box-shadow: 0 0 20px rgba(32,128,208,0.3), 0 0 40px rgba(32,128,208,0.1) !important;
        position: relative;
        overflow: hidden;
    }
    button[kind="primary"]::after, button[kind="primaryFormSubmit"]::after,
    .stButton > button[kind="primary"]::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -60%;
        width: 40%;
        height: 200%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: skewX(-20deg);
        animation: btn-shine 3s infinite;
    }
    button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2068a8 0%, #3090e0 50%, #2068a8 100%) !important;
        box-shadow: 0 0 30px rgba(40,150,230,0.5), 0 0 60px rgba(40,150,230,0.2) !important;
        transform: translateY(-2px);
    }
    /* 次按钮 (secondary + secondaryFormSubmit) */
    button[kind="secondary"], button[kind="secondaryFormSubmit"],
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, rgba(15,35,70,0.9), rgba(20,45,85,0.85)) !important;
        border: 1px solid rgba(0,150,255,0.35) !important;
        color: #b0d0f0 !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 10px rgba(0,120,220,0.12) !important;
    }
    button[kind="secondary"]:hover, button[kind="secondaryFormSubmit"]:hover,
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, rgba(25,50,100,0.9), rgba(30,60,110,0.85)) !important;
        border-color: rgba(0,200,255,0.55) !important;
        color: #d8e8ff !important;
        box-shadow: 0 0 20px rgba(0,160,240,0.28) !important;
    }
    /* 兜底：覆盖 form 内外所有未被 kind 属性匹配到的普通按钮 */
    [data-testid="stFormSubmitButton"] button:not([kind]),
    .stButton > button:not([kind]) {
        background: linear-gradient(135deg, rgba(15,35,70,0.9), rgba(20,45,85,0.85)) !important;
        border: 1px solid rgba(0,150,255,0.35) !important;
        color: #b0d0f0 !important;
    }
    [data-testid="stFormSubmitButton"] button:not([kind]):hover,
    .stButton > button:not([kind]):hover {
        background: linear-gradient(135deg, rgba(25,50,100,0.9), rgba(30,60,110,0.85)) !important;
        border-color: rgba(0,200,255,0.55) !important;
        color: #d8e8ff !important;
    }

    @keyframes btn-shine {
        0% { left: -60%; }
        80% { left: -60%; }
        90% { left: 120%; }
        100% { left: 120%; }
    }

    /* ==================== 进度条 ==================== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0a3d7a, #1e6ab8, #45a0e0, #1e6ab8, #0a3d7a) !important;
        background-size: 200% 100%;
        animation: progress-glow 2s linear infinite;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(30,106,184,0.4);
    }
    .stProgress > div {
        background: rgba(10,30,60,0.5) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0,130,220,0.2);
    }

    @keyframes progress-glow {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    /* ==================== 风险等级卡片 ==================== */
    .risk-card-high {
        background: linear-gradient(135deg, rgba(200,30,40,0.7), rgba(180,15,25,0.75)) !important;
        padding: 36px 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        border: 1px solid rgba(255,100,100,0.4);
        box-shadow: 0 0 30px rgba(255,30,30,0.3), 0 0 60px rgba(255,30,30,0.1);
        animation: pulse-danger 2s ease-in-out infinite;
        backdrop-filter: blur(10px);
        margin-bottom: 8px;
    }
    .risk-card-medium {
        background: linear-gradient(135deg, rgba(200,120,20,0.7), rgba(180,100,10,0.75)) !important;
        padding: 36px 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        border: 1px solid rgba(255,180,40,0.4);
        box-shadow: 0 0 30px rgba(255,140,20,0.3), 0 0 60px rgba(255,140,20,0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 8px;
    }
    .risk-card-low {
        background: linear-gradient(135deg, rgba(20,160,80,0.7), rgba(15,130,60,0.75)) !important;
        padding: 36px 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        border: 1px solid rgba(80,255,140,0.4);
        box-shadow: 0 0 30px rgba(30,200,80,0.3), 0 0 60px rgba(30,200,80,0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 8px;
    }

    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 30px rgba(255,30,30,0.3), 0 0 60px rgba(255,30,30,0.1); }
        50% { box-shadow: 0 0 45px rgba(255,30,30,0.5), 0 0 80px rgba(255,30,30,0.2); }
    }

    /* ==================== 风险因素标签 ==================== */
    .risk-tag-high {
        display: inline-block;
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 0 12px rgba(231,76,60,0.4);
        letter-spacing: 0.5px;
    }
    .risk-tag-medium {
        display: inline-block;
        background: linear-gradient(135deg, #d68910, #f0a020);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 0 10px rgba(240,160,32,0.35);
    }
    .risk-tag-low {
        display: inline-block;
        background: linear-gradient(135deg, #1e8449, #27ae60);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 0 10px rgba(39,174,96,0.35);
    }

    /* ==================== 建议条目 ==================== */
    .suggestion-item {
        background: linear-gradient(135deg, rgba(8,22,50,0.75), rgba(14,30,65,0.65)) !important;
        padding: 14px 18px;
        border-radius: 12px;
        margin: 10px 0;
        border: 1px solid rgba(40,140,210,0.25);
        border-left: 3px solid #40a0e0;
        backdrop-filter: blur(8px);
        box-shadow: 0 0 14px rgba(0,100,200,0.1);
        transition: all 0.3s ease;
    }
    .suggestion-item:hover {
        border-color: rgba(60,180,240,0.45);
        border-left-color: #60c0ff;
        box-shadow: 0 0 22px rgba(0,150,240,0.2);
        transform: translateX(4px);
    }

    /* ==================== 指标卡片 (metric) ==================== */
    .metric-card {
        background: linear-gradient(135deg,
            rgba(6,18,45,0.92), rgba(10,25,58,0.88)) !important;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 2px 18px rgba(0,60,140,0.18);
        border: 1.5px solid rgba(60,170,240,0.35);
        backdrop-filter: blur(10px);
        color: #d0e0ff !important;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(100,200,255,0.55);
        box-shadow: 0 4px 28px rgba(0,120,220,0.28), 0 0 50px rgba(0,140,240,0.08);
    }

    /* ==================== 分割线 ==================== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,
            transparent 0%, rgba(0,150,255,0.25) 20%, rgba(0,200,255,0.4) 50%, rgba(0,150,255,0.25) 80%, transparent 100%) !important;
        margin: 32px 0 !important;
    }

    /* ==================== 表单容器 ==================== */
    [data-testid="stForm"] {
        background: rgba(8,20,50,0.35) !important;
        border: 1.5px solid rgba(60,180,255,0.28) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        backdrop-filter: blur(10px);
    }

    /* ==================== 信息提示框 ==================== */
    .stAlert {
        background: rgba(10,30,60,0.6) !important;
        border: 1px solid rgba(0,150,255,0.2) !important;
        color: #b0d0f0 !important;
        backdrop-filter: blur(8px);
        border-radius: 10px !important;
    }

    /* ==================== 页脚 ==================== */
    footer { visibility: hidden; }
    .stCaption {
        color: #5070a0 !important;
    }

    /* ==================== 动态扫描线装饰 ==================== */
    .scan-line-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .scan-line {
        position: absolute;
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg,
            transparent, rgba(0,200,255,0.08), rgba(0,200,255,0.15), rgba(0,200,255,0.08), transparent);
        animation: scan-down 8s linear infinite;
    }
    .scan-line:nth-child(1) { animation-delay: 0s; top: 10%; }
    .scan-line:nth-child(2) { animation-delay: 2.6s; top: 40%; }
    .scan-line:nth-child(3) { animation-delay: 5.3s; top: 70%; }

    @keyframes scan-down {
        0% { transform: translateY(-100px); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(105vh); opacity: 0; }
    }

    /* ==================== 角落装饰 ==================== */
    .corner-decor-tl, .corner-decor-tr, .corner-decor-bl, .corner-decor-br {
        position: fixed;
        pointer-events: none;
        z-index: 2;
        opacity: 0.5;
    }
    .corner-decor-tl {
        top: 20px; left: 20px;
        width: 60px; height: 60px;
        border-top: 2px solid rgba(0,180,255,0.3);
        border-left: 2px solid rgba(0,180,255,0.3);
    }
    .corner-decor-tr {
        top: 20px; right: 20px;
        width: 60px; height: 60px;
        border-top: 2px solid rgba(0,180,255,0.3);
        border-right: 2px solid rgba(0,180,255,0.3);
    }

    /* ==================== 数据闪烁点 ==================== */
    .data-dots {
        position: fixed;
        top: 50%;
        left: 5%;
        width: 6px;
        height: 6px;
        background: #40b0ff;
        border-radius: 50%;
        box-shadow: 0 0 12px #40b0ff, 0 0 25px rgba(64,176,255,0.5);
        animation: dot-pulse 2s ease-in-out infinite;
        z-index: 0;
        pointer-events: none;
    }
    .data-dots:nth-child(2) {
        top: 20%; left: 90%;
        animation-delay: 0.7s;
    }
    .data-dots:nth-child(3) {
        top: 75%; left: 15%;
        animation-delay: 1.4s;
    }

    @keyframes dot-pulse {
        0%, 100% { opacity: 0.4; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.8); }
    }

    /* ==================== 绘图图表适配 ==================== */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ==================== 响应式微调 ==================== */
    @media (max-width: 768px) {
        .main-title-glow { font-size: 1.8rem; }
    }
</style>

<!-- ═══════════════ 星空 Canvas + 装饰元素 ═══════════════ -->
<canvas id="star-canvas"></canvas>
<div class="scan-line-container">
    <div class="scan-line"></div>
    <div class="scan-line"></div>
    <div class="scan-line"></div>
</div>
<div class="tech-line-top"></div>
<div class="tech-line-bottom"></div>

<script>
(function() {
    var canvas = document.getElementById('star-canvas');
    var ctx = canvas.getContext('2d');
    var stars = [];
    var shootingStars = [];
    var STAR_COUNT = 250;

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // 初始化星星
    for (var i = 0; i < STAR_COUNT; i++) {
        stars.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.8 + 0.3,
            twinkleSpeed: Math.random() * 0.02 + 0.005,
            twinkleOffset: Math.random() * Math.PI * 2,
            brightness: Math.random() * 0.5 + 0.5,
            color: (Math.random() < 0.15)
                ? (Math.random() < 0.5 ? 'rgba(100,200,255,A)' : 'rgba(150,180,255,A)')
                : 'rgba(200,220,255,A)'
        });
    }

    function spawnShootingStar() {
        var startX = Math.random() * canvas.width * 0.8;
        var startY = Math.random() * canvas.height * 0.5;
        shootingStars.push({
            x: startX,
            y: startY,
            len: Math.random() * 80 + 40,
            speed: Math.random() * 8 + 4,
            life: 1.0,
            decay: Math.random() * 0.015 + 0.01
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        var t = Date.now() * 0.001;

        // 绘制静态星星（闪烁）
        for (var i = 0; i < stars.length; i++) {
            var s = stars[i];
            var alpha = s.brightness * (0.6 + 0.4 * Math.sin(t * s.twinkleSpeed * 60 + s.twinkleOffset));
            var color = s.color.replace('A', alpha.toFixed(2));

            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // 亮星添加辉光
            if (s.r > 1.3 && alpha > 0.7) {
                var glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r * 4);
                glow.addColorStop(0, 'rgba(150,200,255,' + (alpha * 0.35).toFixed(2) + ')');
                glow.addColorStop(1, 'rgba(150,200,255,0)');
                ctx.beginPath();
                ctx.arc(s.x, s.y, s.r * 4, 0, Math.PI * 2);
                ctx.fillStyle = glow;
                ctx.fill();
            }
        }

        // 绘制流星
        for (var j = shootingStars.length - 1; j >= 0; j--) {
            var ss = shootingStars[j];
            var grad = ctx.createLinearGradient(
                ss.x, ss.y,
                ss.x - ss.len * 0.6, ss.y - ss.len * 0.3
            );
            grad.addColorStop(0, 'rgba(255,255,255,' + ss.life.toFixed(2) + ')');
            grad.addColorStop(0.2, 'rgba(200,230,255,' + (ss.life * 0.7).toFixed(2) + ')');
            grad.addColorStop(1, 'rgba(200,230,255,0)');

            ctx.beginPath();
            ctx.moveTo(ss.x, ss.y);
            ctx.lineTo(ss.x - ss.len * 0.6, ss.y - ss.len * 0.3);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ss.x -= ss.speed;
            ss.y += ss.speed * 0.5;
            ss.life -= ss.decay;

            if (ss.life <= 0) {
                shootingStars.splice(j, 1);
            }
        }

        // 随机生成流星
        if (Math.random() < 0.015) {
            spawnShootingStar();
        }

        requestAnimationFrame(draw);
    }

    draw();
})();
</script>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║                       中文映射表                             ║
# ╚══════════════════════════════════════════════════════════════╝
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

# ╔══════════════════════════════════════════════════════════════╗
# ║                        加载模型                              ║
# ╚══════════════════════════════════════════════════════════════╝
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

# ╔══════════════════════════════════════════════════════════════╗
# ║                    初始化 session_state                      ║
# ╚══════════════════════════════════════════════════════════════╝
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'show_result' not in st.session_state:
    st.session_state.show_result = False

# ╔══════════════════════════════════════════════════════════════╗
# ║                     🌙 侧边栏                                ║
# ╚══════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="margin: 5px 0; font-family: 'Orbitron', sans-serif; font-size: 1.5rem;">
            ❤️心血管风险预测
        </h2>
        <p style="font-size: 0.7rem; color: #5080a0; letter-spacing: 1px;">CARDIOVASCULAR RISK ML</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. 按步骤填写健康信息
    2. 点击「完成评估」按钮
    3. 查看详细的风险评估结果
    """)

    st.markdown("---")

    st.markdown("### ⚠️ 重要提示")
    st.markdown("本评估仅供健康参考，**不能替代**专业医疗诊断。")

    st.markdown("---")

    st.markdown("### 🎯 风险等级")
    st.markdown("- 🟢 **低风险** (<30%)")
    st.markdown("- 🟡 **中风险** (30-60%)")
    st.markdown("- 🔴 **高风险** (>60%)")

    # 侧边栏底部装饰
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.65rem; color: #4060a0; opacity: 0.6; padding-top: 10px;">
        &ensp;◆&ensp;活得长长久久小组&ensp;◆&ensp;
    </div>
    """, unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║                     🔷 主页面标题                            ║
# ╚══════════════════════════════════════════════════════════════╝
col_title1, col_title2, col_title3 = st.columns([1, 3, 1])
with col_title2:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 15px 0;">
        <div class="main-title-glow">❤️ 心血管疾病风险预测系统</div>
        <p class="subtitle-glow">基于机器学习的心脏健康评估工具</p>
        <div style="margin-top: 8px;">
            <span style="display: inline-block; width: 40px; height: 2px; background: linear-gradient(90deg, transparent, #4090d0, transparent); margin: 0 8px;"></span>
            <span style="font-size: 0.65rem; color: #4060a0; letter-spacing: 2px;">HEALTH · FUTURE</span>
            <span style="display: inline-block; width: 40px; height: 2px; background: linear-gradient(90deg, transparent, #4090d0, transparent); margin: 0 8px;"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if model is None:
    st.stop()

# ╔══════════════════════════════════════════════════════════════╗
# ║                     反向映射表                               ║
# ╚══════════════════════════════════════════════════════════════╝
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

# ╔══════════════════════════════════════════════════════════════╗
# ║                    步骤定义 (4步, 25字段)                    ║
# ╚══════════════════════════════════════════════════════════════╝
step_fields = {
    1: {
        "title": "基本信息",
        "icon": "📊",
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


def get_risk_level(inputs):
    risk_factors = []

    smoker = inputs.get('SmokerStatus', '')
    if smoker in ["Current smoker - now smokes every day", "Current smoker - now smokes some days"]:
        risk_factors.append({"因素": "吸烟", "等级": "高", "说明": "吸烟会损伤血管内壁，增加血栓风险"})
    elif smoker == "Former smoker":
        risk_factors.append({"因素": "吸烟史", "等级": "中", "说明": "既往吸烟史仍有影响，但戒烟会逐步降低风险"})

    bmi = inputs.get('BMI', 22)
    if bmi >= 28:
        risk_factors.append({"因素": "肥胖", "等级": "高", "说明": "肥胖增加心脏负担，可能导致高血压、高血脂"})
    elif bmi >= 24:
        risk_factors.append({"因素": "超重", "等级": "中", "说明": "超重会增加心血管疾病风险"})

    diabetes = inputs.get('HadDiabetes', '')
    if diabetes == "Yes":
        risk_factors.append({"因素": "糖尿病", "等级": "高", "说明": "糖尿病会损害血管，增加心血管疾病风险"})
    elif diabetes == "No, pre-diabetes or borderline diabetes":
        risk_factors.append({"因素": "糖尿病前期", "等级": "中", "说明": "糖尿病前期需要警惕，建议控制饮食"})

    if inputs.get('HadAngina') == "Yes":
        risk_factors.append({"因素": "心绞痛史", "等级": "高", "说明": "有心绞痛史说明心脏已存在供血问题"})

    if inputs.get('HadStroke') == "Yes":
        risk_factors.append({"因素": "中风史", "等级": "高", "说明": "中风与心血管疾病有共同的病理基础"})

    if inputs.get('PhysicalActivities') == "No":
        risk_factors.append({"因素": "缺乏运动", "等级": "中", "说明": "缺乏运动会导致心肺功能下降"})

    sleep_hours = inputs.get('SleepHours', 7)
    if sleep_hours < 6:
        risk_factors.append({"因素": "睡眠不足", "等级": "中", "说明": "睡眠不足会增加炎症反应和血压"})
    elif sleep_hours > 9:
        risk_factors.append({"因素": "睡眠过长", "等级": "低", "说明": "睡眠时间过长也可能与心脏问题相关"})

    if inputs.get('AlcoholDrinkers') == "Yes":
        risk_factors.append({"因素": "饮酒", "等级": "中", "说明": "过量饮酒会增加心血管疾病风险"})

    if inputs.get('HadDepressiveDisorder') == "Yes":
        risk_factors.append({"因素": "抑郁症", "等级": "中", "说明": "抑郁症与心血管疾病风险相关"})

    if inputs.get('HadKidneyDisease') == "Yes":
        risk_factors.append({"因素": "慢性肾病", "等级": "高", "说明": "肾病与心血管疾病密切相关"})

    return risk_factors


# ╔══════════════════════════════════════════════════════════════╗
# ║                     📋 表单处理                               ║
# ╚══════════════════════════════════════════════════════════════╝
if not st.session_state.show_result:
    step_info = step_fields[st.session_state.step]

    st.markdown(f"""
    <div class="step-header-card">
        <h2 style="margin:0;">{step_info['icon']} {step_info['title']}</h2>
        <p style="margin:5px 0 0 0;">📌 {step_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 3, 1])
    with cols[1]:
        st.markdown(f"<p style='text-align:center; font-size:0.9rem; color: #8090b0; padding: 6px 0;'>第 {st.session_state.step}/4 步</p>", unsafe_allow_html=True)
        st.progress(st.session_state.step / 4)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with st.form(key=f"step_{st.session_state.step}"):
        field_values = {}

        form_cols = st.columns(2)

        for idx, (field_key, field_label, field_type, field_options) in enumerate(step_info["fields"]):
            with form_cols[idx % 2]:
                if field_type == "select":
                    current_val = st.session_state.inputs.get(field_key)
                    if current_val:
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

        if st.session_state.step == 1:
            height = field_values.get("HeightInMeters", 1.7)
            weight = field_values.get("WeightInKilograms", 70.0)
            bmi = weight / (height ** 2)
            field_values["BMI"] = bmi

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.session_state.step > 1:
                if st.form_submit_button("← 上一步", use_container_width=True, type="secondary"):
                    st.session_state.step -= 1
                    st.rerun()

        with col2:
            if st.form_submit_button("保存并继续 →", type="primary", use_container_width=True):
                for key, value in field_values.items():
                    if key == "BMI":
                        st.session_state.inputs[key] = value
                    else:
                        reverse_map = reverse_maps.get(key, {})
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
            if st.form_submit_button("重新开始", use_container_width=True, type="secondary"):
                st.session_state.inputs = {}
                st.session_state.step = 1
                st.session_state.show_result = False
                st.rerun()

# ╔══════════════════════════════════════════════════════════════╗
# ║                     📈 结果显示                              ║
# ╚══════════════════════════════════════════════════════════════╝
if st.session_state.show_result:
    inputs = st.session_state.inputs

    if len(inputs) < 20:
        st.warning("⚠️ 数据不完整，请重新填写")
        if st.button("重新开始评估"):
            st.session_state.inputs = {}
            st.session_state.step = 1
            st.session_state.show_result = False
            st.rerun()
        st.stop()

    input_df = pd.DataFrame([inputs])

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

    input_df = input_df[[col for col in features if col in input_df.columns]]

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[features]
    input_df = input_df.fillna(0)

    risk_prob = model.predict_proba(input_df)[0][1]
    risk_percent = risk_prob * 100

    bmi = inputs.get('BMI', 24)
    bmi_status = "正常" if 18.5 <= bmi < 24 else "偏瘦" if bmi < 18.5 else "超重" if bmi < 28 else "肥胖"
    sleep_status = "良好" if 7 <= inputs.get('SleepHours', 7) <= 8 else "不足" if inputs.get('SleepHours', 7) < 7 else "过多"
    activity_status = "活跃" if inputs.get('PhysicalActivities') == "Yes" else "缺乏运动"

    smoker_val = inputs.get('SmokerStatus', '')
    smoker_cn = ""
    for cn, en in SMOKER_CN.items():
        if en == smoker_val:
            smoker_cn = cn
            break

    st.markdown("## 📈 评估结果")

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

    st.markdown("### 📊 健康指标概览")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6090c0;">风险评分</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #d0e8ff;">{risk_percent:.1f}%</div>
            <div style="font-size: 0.8rem; color: {'#ff6b6b' if risk_percent>=60 else '#ffa502' if risk_percent>=30 else '#2ed573'};">
                {'🔴 高风险' if risk_percent>=60 else '🟡 中风险' if risk_percent>=30 else '🟢 低风险'}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col_m2:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6090c0;">BMI指数</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #d0e8ff;">{bmi:.1f}</div>
            <div style="font-size: 0.8rem; color: #80a0c0;">{bmi_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_m3:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6090c0;">睡眠质量</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #d0e8ff;">{inputs.get('SleepHours', 7)}h</div>
            <div style="font-size: 0.8rem; color: #80a0c0;">{sleep_status}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_m4:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #6090c0;">运动情况</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #d0e8ff;">{activity_status}</div>
            <div style="font-size: 0.8rem; color: #80a0c0;">{'✅ 每周运动' if activity_status=='活跃' else '⚠️ 建议增加运动'}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("### 📋 风险因素分析")

    risk_factors = get_risk_level(inputs)

    if risk_factors:
        for rf in risk_factors:
            level = rf["等级"]
            if level == "高":
                badge = '<span class="risk-tag-high">高风险</span>'
            elif level == "中":
                badge = '<span class="risk-tag-medium">中风险</span>'
            else:
                badge = '<span class="risk-tag-low">低风险</span>'
            st.markdown(f"""
            <div class="suggestion-item">
                {badge} <strong>{rf['因素']}</strong>&emsp;{rf['说明']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 未检测到显著风险因素，请继续保持健康生活！")

    st.markdown("---")

    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        if st.button("🔄 重新开始评估", use_container_width=True, type="primary"):
            st.session_state.inputs = {}
            st.session_state.step = 1
            st.session_state.show_result = False
            st.rerun()

# ╔══════════════════════════════════════════════════════════════╗
# ║                     页脚                                     ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("---")
st.caption("⚠️ 免责声明：本工具仅供健康参考，不构成医疗建议。如有心脏不适症状，请立即就医。")
st.caption("💡 本系统基于机器学习算法，评估结果仅供参考")
