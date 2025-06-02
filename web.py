import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import os
from collections import Counter

# -------------------- 1. 页面设置 --------------------
st.set_page_config(page_title="抗菌肽预测系统", page_icon="🧪", layout="wide")
st.title("抗菌肽预测系统")

# -------------------- 2. 左侧功能导航 --------------------
page = st.sidebar.radio(
    "请选择功能页面：",
    ("抗菌肽预测", "类别预测", "MIC值预测")
)

# -------------------- 3. 抗菌肽预测页面 --------------------
if page == "抗菌肽预测":
    st.subheader("抗菌肽预测")
    st.write("请在下方输入氨基酸序列，我们将进行AAC分析，并预测是否为抗菌肽。")

    # -------------------- 4. 加载模型和标准化器 --------------------
    # 获取当前文件目录
    BASE_DIR = os.path.dirname(__file__)
    # 使用相对路径加载模型
    MODEL_PATH = os.path.join(BASE_DIR, "models", "antimicrobial_peptide_model_improved.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_improved.pkl")

    # 加载模型和标准化器
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # -------------------- 5. AAC 特征提取函数 --------------------
    def compute_aac(sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        count = Counter(sequence)
        seq_len = len(sequence)
        return [count[aa] / seq_len if seq_len > 0 else 0 for aa in AA]

    # -------------------- 6. 上传 FASTA 批量预测 --------------------
    uploaded_file = st.file_uploader("上传包含肽序列的 FASTA 文件", type=["fasta", "txt"])

    if uploaded_file:
        fasta_sequences = []
        for line in uploaded_file:
            line = line.decode('utf-8').strip()
            if not line.startswith('>'):
                fasta_sequences.append(line)

        # AAC 分析
        aac_features = [compute_aac(seq) for seq in fasta_sequences]
        df = pd.DataFrame(aac_features, columns=list('ACDEFGHIKLMNPQRSTVWY'))

        # 标准化 + 预测
        df_scaled = scaler.transform(df)
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        # 展示结果
        result_df = pd.DataFrame({
            "序列": fasta_sequences,
            "预测类别": ["抗菌肽" if p == 1 else "非抗菌肽" for p in predictions],
            "预测概率": probabilities
        })

        st.write("预测结果：", result_df)

        # 下载结果
        output = io.BytesIO()
        result_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        st.download_button("下载预测结果", data=output, file_name="抗菌肽预测结果.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("请上传 FASTA 文件")

# -------------------- 8. 类别预测页面 --------------------
elif page == "类别预测":
    st.subheader("类别预测")
    st.info("此功能开发中，将用于细菌种类分类（如 A. baumannii, P. aeruginosa 等）。")

