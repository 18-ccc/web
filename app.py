import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# 载入训练好的SVM模型和标准化器
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')


# 计算AAC特征的函数
def extract_aac_features(sequence):
    """
    从给定的蛋白质序列提取AAC（氨基酸组成）特征。

    参数:
    sequence (str): 蛋白质序列（例如：'ARNDCEQGH'）

    返回:
    features_df (DataFrame): 包含AAC特征的DataFrame，每个氨基酸的频率作为一个特征
    """
    amino_acids = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ]

    sequence = sequence.upper()  # 转为大写以统一处理
    total_length = len(sequence)

    # 创建一个字典，保存每种氨基酸的计数
    aa_counts = {aa: 0 for aa in amino_acids}

    # 遍历序列，统计每种氨基酸的出现次数
    for aa in sequence:
        if aa in aa_counts:
            aa_counts[aa] += 1

    # 计算每种氨基酸的频率
    aa_frequencies = {aa: count / total_length for aa, count in aa_counts.items()}

    # 转换为DataFrame
    features_df = pd.DataFrame([aa_frequencies])

    return features_df


# Streamlit 应用逻辑
st.title("蛋白质序列特征提取与预测")

# 输入框：让用户输入蛋白质序列
sequence = st.text_area("请输入蛋白质序列", "")

# 按钮：点击后提取特征并进行预测
if st.button("提取特征并预测"):
    if sequence:
        # 提取AAC特征
        features_df = extract_aac_features(sequence)

        # 显示提取的特征
        st.write("提取的AAC特征:")
        st.write(features_df)

        # 标准化特征
        features_scaled = scaler.transform(features_df)

        # 使用SVM模型进行预测
        prediction = svm_model.predict(features_scaled)

        # 显示预测结果
        st.write(f"预测结果: {'抗菌肽' if prediction[0] == 1 else '非抗菌肽'}")
    else:
        st.error("请输入有效的蛋白质序列！")
