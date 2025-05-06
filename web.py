import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
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

    # 模型路径（使用改进版模型）
    MODEL_PATH = r"D:\bishe\毕业设计\分类模型\antimicrobial_peptide_model_improved.pkl"
    SCALER_PATH = r"D:\bishe\毕业设计\分类模型\scaler_improved.pkl"

    # 加载模型
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # AAC 特征提取函数
    def compute_aac(sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        count = Counter(sequence)
        seq_len = len(sequence)
        return [count[aa] / seq_len if seq_len > 0 else 0 for aa in AA]

    # 上传 FASTA 批量预测
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

        # 下载
        output = io.BytesIO()
        result_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        st.download_button("下载预测结果", data=output, file_name="抗菌肽预测结果.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("请上传 FASTA 文件，或者手动输入氨基酸序列。")

    # 手动输入序列预测
    st.subheader("手动输入氨基酸序列进行预测")
    input_sequence = st.text_input("请输入氨基酸序列（单字母代码）:")

    if st.button("预测"):
        if input_sequence:
            input_aac = np.array(compute_aac(input_sequence)).reshape(1, -1)
            input_scaled = scaler.transform(input_aac)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0, 1]

            if prediction == 1:
                st.success(f"预测结果：**抗菌肽** (概率: {probability:.2f}) 🦠")
            else:
                st.error(f"预测结果：**非抗菌肽** (概率: {probability:.2f}) ❌")
        else:
            st.warning("请输入有效的氨基酸序列！")

# -------------------- 4. 类别预测页面 --------------------
elif page == "类别预测":
    st.subheader("类别预测")
    st.info("此功能开发中，将用于细菌种类分类（如 A. baumannii, P. aeruginosa 等）。")

# -------------------- 5. MIC 值预测页面 --------------------
elif page == "MIC值预测":
    st.subheader("MIC值预测")
    st.info("此功能开发中，将用于回归预测抗菌肽的最小抑菌浓度（MIC）。")
