import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score
)

# ====== 1. 加载并合并特征文件 ======
feature_files = ["apaac.xlsx", "paac.xlsx", "qso.xlsx", "aac.xlsx"]
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"

dfs = []
for file in feature_files:
    path = os.path.join(base_path, file)
    df = pd.read_excel(path)
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on=["SampleName", "label"], how="inner")

print("合并后的维度：", merged.shape)

# ====== 2. 特征与标签准备 ======
X = merged.drop(["SampleName", "label"], axis=1, errors="ignore")
y = merged["label"].reset_index(drop=True)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

if X.shape[1] == 0:
    raise ValueError("特征为空，请检查特征文件内容。")

# ====== 3. 不进行特征选择 ======

# ====== 4. 划分训练集和测试集 + 标准化 ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====== 5. 定义 KNN 模型 ======
knn_model = KNeighborsClassifier(
    n_neighbors=5,  # 可调节K值
    weights='uniform',  # 所有邻居权重相同
    metric='minkowski',  # 默认欧氏距离
    p=2,
    n_jobs=-1
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

        # KNN predict_proba 有可能不存在，当不存在时用决策函数或跳过AUC
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val_cv)[:, 1]
            auc_score = roc_auc_score(y_val_cv, probs)
        else:
            auc_score = np.nan

        accs.append(accuracy_score(y_val_cv, preds))
        recalls.append(recall_score(y_val_cv, preds, average='weighted'))
        f1s.append(f1_score(y_val_cv, preds, average='weighted'))
        aucs.append(auc_score)

    # 处理AUC中nan的情况
    aucs = [a for a in aucs if not np.isnan(a)]
    auc_mean = np.mean(aucs) if aucs else float('nan')
    auc_std = np.std(aucs) if aucs else float('nan')

    return (
        np.mean(accs), np.std(accs),
        np.mean(recalls), np.std(recalls),
        np.mean(f1s), np.std(f1s),
        auc_mean, auc_std
    )


# ====== 7. 执行训练集交叉验证评估 ======
acc_mean, acc_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std = \
    cross_validate_metrics(knn_model, X_train_scaled, y_train)

# ====== 8. 测试集评估 ======
knn_model.fit(X_train_scaled, y_train)
test_preds = knn_model.predict(X_test_scaled)
if hasattr(knn_model, "predict_proba"):
    test_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
else:
    test_auc = float('nan')

test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')

# ====== 9. 打印评估结果 ======
print(f"\n========== 合并特征文件：{', '.join(feature_files)} ==========")
print(f"【训练集交叉验证】")
print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Recall:   {rec_mean:.4f} ± {rec_std:.4f}")
print(f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"AUC:      {auc_mean if not np.isnan(auc_mean) else 'N/A'} ± {auc_std if not np.isnan(auc_std) else 'N/A'}")
print(f"\n【测试集】")
print(f"Accuracy: {test_acc:.4f}")
print(f"Recall:   {test_rec:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"AUC:      {test_auc if not np.isnan(test_auc) else 'N/A'}")
print("===================================")
