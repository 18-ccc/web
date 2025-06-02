import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import os
from collections import Counter

# -------------------- 1. 页面设置 --------------------
st.set_page_config(page_title="抗菌肽预测系统", page_icon="🧪", layout="wide")

# 自定义样式，缩小侧边栏和主内容区的间距
st.markdown(
    """
    <style>
    /* 缩小主内容区左右内边距 */
    .css-1d391kg {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* 缩小侧边栏左右内边距 */
    .css-1v3fvcr {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* 按钮样式 */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 18px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }

    /* 文件上传标签样式 */
    .stFileUploader > div > label > div {
        font-weight: bold;
        color: #4CAF50;
        font-size: 18px;
        margin-bottom: 10px;
    }

    /* 主标题样式 */
    h1 {
        text-align: center;
        color: #4CAF50;
        font-size: 48px;
        margin-bottom: 10px;
    }
    hr {
        border:1px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题，带图标
st.markdown(
    """
    <h1>🧪 抗菌肽预测系统</h1>
    <hr>
    """,
    unsafe_allow_html=True
)

# 侧边栏功能选择，带图标
page = st.sidebar.radio(
    "请选择功能页面：",
    ("🧬 抗菌肽预测", "🦠 类别预测"),
    index=0,
    help="选择不同的功能模块"
)

if page == "🧬 抗菌肽预测":
    st.subheader("🧬 抗菌肽预测")
    st.write("请上传或输入氨基酸序列（单字母代码），系统将自动进行AAC分析并预测是否为抗菌肽。")

    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "models", "antimicrobial_peptide_model_improved.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_improved.pkl")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    def compute_aac(sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        count = Counter(sequence.upper())
        seq_len = len(sequence)
        return [count[aa] / seq_len if seq_len > 0 else 0 for aa in AA]

    uploaded_file = st.file_uploader("上传包含肽序列的 FASTA 文件", type=["fasta", "txt"])

    if uploaded_file:
        fasta_sequences = []
        for line in uploaded_file:
            line = line.decode('utf-8').strip()
            if not line.startswith('>') and line != '':
                fasta_sequences.append(line)

        if len(fasta_sequences) == 0:
            st.warning("未检测到有效肽序列，请检查文件格式。")
        else:
            with st.spinner("正在分析和预测，请稍候..."):
                aac_features = [compute_aac(seq) for seq in fasta_sequences]
                df = pd.DataFrame(aac_features, columns=list('ACDEFGHIKLMNPQRSTVWY'))

                df_scaled = scaler.transform(df)
                predictions = model.predict(df_scaled)
                probabilities = model.predict_proba(df_scaled)[:, 1]

                result_df = pd.DataFrame({
                    "序列": fasta_sequences,
                    "预测类别": ["抗菌肽" if p == 1 else "非抗菌肽" for p in predictions],
                    "预测概率": probabilities.round(4)
                })

                st.success("预测完成！")
                st.dataframe(result_df, use_container_width=True)

                output = io.BytesIO()
                result_df.to_excel(output, index=False, engine='openpyxl')
                output.seek(0)
                st.download_button(
                    "⬇️ 下载预测结果",
                    data=output,
                    file_name="抗菌肽预测结果.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("请上传 FASTA 文件进行批量预测。")

elif page == "🦠 类别预测":
    st.subheader("🦠 类别预测")
    st.info("此功能开发中，未来将支持细菌种类分类（如 A. baumannii、P. aeruginosa 等）。")
