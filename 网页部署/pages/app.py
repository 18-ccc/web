import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
import io
from collections import Counter

# -------------------- 1. 设置页面 --------------------
st.set_page_config(page_title="抗菌肽预测系统", page_icon="🧪", layout="wide")
st.title("抗菌肽预测系统")

# -------------------- 2. 视频展示 --------------------
video_path = r"D:\HuaweiMoveData\Users\陈雯静.LAPTOP-CJOIH1UC\Desktop\WeChat_20250304132551.mp4"

# 读取视频文件并进行 Base64 编码
with open(video_path, "rb") as video_file:
    video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode()

# 使用 HTML 代码嵌入视频，并隐藏进度条
video_html = f"""
    <video width="800" height="450" autoplay loop muted playsinline controlslist="nodownload nofullscreen noremoteplayback" style="outline: none;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        您的浏览器不支持视频播放
    </video>
"""
st.markdown(video_html, unsafe_allow_html=True)

# -------------------- 3. 加载模型 --------------------
MODEL_PATH = r"/分类模型/antimicrobial_peptide_model.pkl"
SCALER_PATH = r"/分类模型/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.write("请在下方输入氨基酸序列，我们将进行AAC分析，并预测是否为抗菌肽。")

# -------------------- 4. AAC 分析函数 --------------------
def compute_aac(sequence):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    count = Counter(sequence)
    seq_len = len(sequence)
    return [count[aa] / seq_len if seq_len > 0 else 0 for aa in AA]

# -------------------- 5. 上传 FASTA 文件进行批量预测 --------------------
uploaded_file = st.file_uploader("上传包含肽序列的 FASTA 文件", type=["fasta", "txt"])

if uploaded_file:
    fasta_sequences = []
    for line in uploaded_file:
        line = line.decode('utf-8').strip()
        if not line.startswith('>'):
            fasta_sequences.append(line)

    # 进行 AAC 分析
    aac_features = [compute_aac(seq) for seq in fasta_sequences]
    df = pd.DataFrame(aac_features, columns=list('ACDEFGHIKLMNPQRSTVWY'))

    # 数据标准化
    df_scaled = scaler.transform(df)

    # 预测
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    # 生成结果 DataFrame
    result_df = pd.DataFrame({
        "序列": fasta_sequences,
        "预测类别": ["抗菌肽" if p == 1 else "非抗菌肽" for p in predictions],
        "预测概率": probabilities
    })

    st.write("预测结果：", result_df)

    # 提供下载
    output = io.BytesIO()
    result_df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    st.download_button(label="下载预测结果", data=output, file_name="抗菌肽预测结果.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("请上传 FASTA 文件，或者手动输入氨基酸序列。")

# -------------------- 6. 手动输入氨基酸序列进行预测 --------------------
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
