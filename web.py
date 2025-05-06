import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from Bio import SeqIO
import io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)

# 页面设置
st.set_page_config(page_title="抗菌肽预测", layout="wide")
st.title("🧬 抗菌肽预测系统")
st.markdown("上传 FASTA 文件或输入一个序列，系统将提取 AAC 特征并进行抗菌肽预测。")

# AAC 特征计算函数
def compute_aac(sequence):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    count = Counter(sequence)
    seq_len = len(sequence)
    return [count[aa] / seq_len if seq_len > 0 else 0 for aa in AA]

# 上传 FASTA 文件或输入序列
fasta_file = st.file_uploader("📁 上传 FASTA 文件", type=["fasta", "fa"])
sequence_input = st.text_area("🔤 输入一个序列进行预测", placeholder="输入一个单独的氨基酸序列")

# 处理上传的文件或输入的序列
sequences = []

if fasta_file is not None:
    st.info("✅ 开始读取序列并提取 AAC 特征...")
    for record in SeqIO.parse(io.StringIO(fasta_file.getvalue().decode()), "fasta"):
        seq = str(record.seq).upper()
        aac = compute_aac(seq)
        sequences.append(aac)

elif sequence_input:
    st.info("✅ 开始处理输入的单个序列...")
    seq = sequence_input.strip().upper()
    aac = compute_aac(seq)
    sequences.append(aac)

if sequences:
    # 构建数据框并训练模型
    df = pd.DataFrame(sequences, columns=list("ACDEFGHIKLMNPQRSTVWY"))

    # 特征提取与建模
    X = df

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X, X)  # 训练时不需要标签，但需要根据自己的需求调整

    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_10_features = feature_importances.nlargest(10).index.tolist()
    X_selected = X[top_10_features]

    X_train, X_test = train_test_split(X_selected, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_split=10,
        min_samples_leaf=4, class_weight='balanced', random_state=42
    )
    best_rf_model.fit(X_train_scaled, X_train)

    # 模型评估
    st.subheader("📊 模型评估结果")
    st.write(f"测试集 Accuracy: {best_rf_model.score(X_test_scaled, X_test):.4f}")

    st.subheader("📌 Precision / Recall / F1")
    st.json({
        "Precision": precision_score(X_test, best_rf_model.predict(X_test), average='weighted'),
        "Recall": recall_score(X_test, best_rf_model.predict(X_test), average='weighted'),
        "F1-score": f1_score(X_test, best_rf_model.predict(X_test), average='weighted'),
    })

    st.subheader("🧩 混淆矩阵")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(X_test, best_rf_model.predict(X_test)), annot=True, fmt="d", cmap="Oranges", ax=ax)
    ax.set_title("Test Confusion Matrix")
    st.pyplot(fig)

    st.write("---")
    st.subheader("🔮 重新上传新 FASTA 文件进行预测")
    pred_file = st.file_uploader("上传新的 FASTA 文件进行抗菌肽预测", type=["fasta", "fa"], key="predict_fasta")

    def predict_sequence(sequence, model, scaler, selected_features):
        aac = compute_aac(sequence)
        df = pd.DataFrame([aac], columns=list("ACDEFGHIKLMNPQRSTVWY"))
        df_selected = df[selected_features]
        df_scaled = scaler.transform(df_selected)
        prob = model.predict_proba(df_scaled)[0][1]
        pred = model.predict(df_scaled)[0]
        return pred, prob

    if pred_file is not None:
        st.info("正在分析并预测新序列...")
        sequences = []
        for record in SeqIO.parse(io.StringIO(pred_file.getvalue().decode()), "fasta"):
            seq = str(record.seq).upper()
            pred, prob = predict_sequence(seq, best_rf_model, scaler, top_10_features)
            sequences.append({"Name": record.id, "Prediction": pred, "Probability": prob})
        result_df = pd.DataFrame(sequences)
        st.dataframe(result_df)
        st.download_button("📥 下载预测结果", result_df.to_csv(index=False), file_name="prediction_results.csv")

else:
    st.info("请上传 FASTA 文件或输入一个氨基酸序列进行预测。")
