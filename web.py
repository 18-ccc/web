import streamlit as st
import pandas as pd
import io
import random

# 页面设置与样式
st.set_page_config(page_title="抗菌肽预测系统", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    h1 { text-align: center; color: #4CAF50; font-size: 48px; margin-bottom: 10px; }
    .stButton>button {
        background-color: #4CAF50; color: white; font-weight: bold; border-radius: 8px;
        padding: 8px 18px; transition: background-color 0.3s ease;
    }
    .stButton>button:hover { background-color: #45a049; }
    hr { border: 1px solid #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# 功能选择
page = st.sidebar.radio(
    "请选择功能页面：",
    ("🧬 抗菌肽预测", "🧫 类别预测"),
    index=0
)

# 动态主标题
if page == "🧬 抗菌肽预测":
    main_title = "🧪 抗菌肽预测系统"
else:
    main_title = "🧫 碳青霉烯类耐药性抗菌肽的预测"

st.markdown(f"<h1>{main_title}</h1><hr>", unsafe_allow_html=True)


# 模拟预测函数（抗菌肽预测）
def fake_predict(sequences):
    result = []
    for seq in sequences:
        prob = round(random.uniform(0, 1), 4)
        label = "抗菌肽" if prob > 0.5 else "非抗菌肽"
        result.append({"序列": seq, "预测概率": prob})
    return pd.DataFrame(result)


# 模拟细菌分类预测函数
def fake_bacteria_predict(sequences, bacteria_type):
    result = []
    for seq in sequences:
        # 随机生成预测概率
        prob = round(random.uniform(0, 1), 4)
        # 根据选择的菌种进行不同的预测阈值设置
        if bacteria_type == "鲍曼不动杆菌":
            label = "敏感" if prob > 0.7 else "耐药"
        elif bacteria_type == "肠杆菌科":
            label = "敏感" if prob > 0.6 else "耐药"
        else:  # 铜绿假单胞菌
            label = "敏感" if prob > 0.65 else "耐药"

        # 判断是否为目标菌种
        is_target_bacteria = "是" if label == "敏感" else "否"

        result.append({"序列": seq, "预测概率": prob, "是否目标菌种": is_target_bacteria})
    return pd.DataFrame(result)


# 上传并预测
def upload_and_predict(label="抗菌肽预测", bacteria_type=None):
    st.subheader(f"📥 {label} - 上传肽序列")
    uploaded_file = st.file_uploader("上传 FASTA 文件（支持 .fasta 或 .txt）", type=["fasta", "txt"])

    if uploaded_file:
        fasta_sequences = []
        for line in uploaded_file:
            line = line.decode("utf-8").strip()
            if not line.startswith(">") and line != "":
                fasta_sequences.append(line)

        if not fasta_sequences:
            st.warning("未检测到有效肽序列，请检查格式。")
        else:
            st.info("⚙️ 预测中")
            # 如果是细菌类别预测，调用 fake_bacteria_predict
            if bacteria_type:
                result_df = fake_bacteria_predict(fasta_sequences, bacteria_type)
            else:
                result_df = fake_predict(fasta_sequences)

            # 显示结果表格
            st.dataframe(result_df, use_container_width=True)

            # 下载按钮
            output = io.BytesIO()
            result_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(
                "⬇️ 下载预测结果",
                data=output,
                file_name="预测结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("请上传包含肽序列的文件。")


# 页面路由
if page == "🧬 抗菌肽预测":
    upload_and_predict("抗菌肽预测")
elif page == "🧫 类别预测":
    target = st.selectbox("选择预测目标细菌：", ["鲍曼不动杆菌", "肠杆菌科", "铜绿假单胞菌"])
    upload_and_predict(f"{target} - 类别预测", bacteria_type=target)
