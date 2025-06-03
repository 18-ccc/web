import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt

# ====== 1. 加载并合并特征文件 ======
feature_files = ["apaac.xlsx", "paac.xlsx", "qso.xlsx", "aac.xlsx"]
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"

dfs = []
for file in feature_files:
    path = os.path.join(base_path, file)
    df = pd.read_excel(path)
    dfs.append(df)

# 多个特征数据表按 SampleName 和 label 内连接合并
merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on=["SampleName", "label"], how="inner")

print("合并后的维度：", merged.shape)

# ====== 2. 特征与标签准备 ======
X = merged.drop(["SampleName", "label"], axis=1, errors="ignore")  # 去除非数值列
y = merged["label"].reset_index(drop=True)
X = X.apply(pd.to_numeric, errors="coerce")  # 保证所有特征为数值型
X = X.fillna(X.mean())  # 用均值填补缺失值

if X.shape[1] == 0:
    raise ValueError("特征为空，请检查特征文件内容。")

# ====== 3. 不进行特征选择，直接使用全部特征 ======

# ====== 4. 划分训练集和测试集 + 标准化 ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====== 5. 定义逻辑回归模型 ======
logistic_model = LogisticRegression(
    solver="liblinear",  # 适合小数据集，支持 L1/L2 正则
    penalty="l2",        # 使用 L2 正则
    C=1.0,               # 正则化强度
    class_weight="balanced",  # 自动处理类别不平衡
    random_state=42
)

# ====== 6. 自定义函数：手动交叉验证评估指标 ======
def cross_validate_metrics(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, recalls, f1s, aucs = [], [], [], []

    for train_idx, valid_idx in skf.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[valid_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train_cv, y_train_cv)
        preds = model.predict(X_val_cv)
        probs = model.predict_proba(X_val_cv)[:, 1]

        accs.append(accuracy_score(y_val_cv, preds))
        recalls.append(recall_score(y_val_cv, preds, average='weighted'))
        f1s.append(f1_score(y_val_cv, preds, average='weighted'))
        aucs.append(roc_auc_score(y_val_cv, probs))

    return (
        np.mean(accs), np.std(accs),
        np.mean(recalls), np.std(recalls),
        np.mean(f1s), np.std(f1s),
        np.mean(aucs), np.std(aucs)
    )

# ====== 7. 执行训练集交叉验证评估 ======
acc_mean, acc_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std = \
    cross_validate_metrics(logistic_model, X_train_scaled, y_train)

# ====== 8. 测试集评估 ======
logistic_model.fit(X_train_scaled, y_train)
test_preds = logistic_model.predict(X_test_scaled)
test_probs = logistic_model.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_auc = roc_auc_score(y_test, test_probs)

# ====== 9. 打印评估结果 ======
print(f"\n========== 合并特征文件：{', '.join(feature_files)} ==========")
print(f"【训练集交叉验证】")
print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Recall:   {rec_mean:.4f} ± {rec_std:.4f}")
print(f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
print(f"\n【测试集】")
print(f"Accuracy: {test_acc:.4f}")
print(f"Recall:   {test_rec:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"AUC:      {test_auc:.4f}")
print("===================================")
