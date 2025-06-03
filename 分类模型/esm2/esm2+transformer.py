import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from collections import Counter

# ===== Transformer 编码模块 =====
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch_size, d_model]
        return x

# ===== 加载两个 ESM2 特征并拼接 =====
def load_dual_esm2_features(feature_dir1, feature_dir2, label_dict):
    all_features = []
    labels = []
    not_found = []

    for seq_id, label in label_dict.items():
        path1 = os.path.join(feature_dir1, f"{seq_id}.npy")
        path2 = os.path.join(feature_dir2, f"{seq_id}.npy")

        if not os.path.exists(path1) or not os.path.exists(path2):
            not_found.append(seq_id)
            continue

        feat1 = np.load(path1)  # shape: [L1, D1]
        feat2 = np.load(path2)  # shape: [L2, D2]

        # 对齐长度
        min_len = min(feat1.shape[0], feat2.shape[0])
        feat1 = feat1[:min_len]
        feat2 = feat2[:min_len]

        feat = np.concatenate([feat1, feat2], axis=1)  # shape: [min_len, D1+D2]
        all_features.append(feat)
        labels.append(label)

    print(f"\n加载成功样本数: {len(labels)}, 缺失样本数: {len(not_found)}")
    return all_features, np.array(labels)

# ===== 模型评估函数 =====
def evaluate_model(X, y, name="模型"):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\n[{name}] SMOTE 前类别分布: {Counter(y)}")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
    print(f"[{name}] SMOTE 后类别分布: {Counter(y_resampled)}")

    # ============ 交叉验证 ============

    print(f"\n[{name}] 训练集交叉验证评估:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, recalls, f1s, aucs = [], [], [], []

    for train_idx, val_idx in skf.split(X_resampled, y_resampled):
        X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
        y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

        clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_prob))

    print("=== Cross-Validation on Training Set ===")
    print(f"Accuracy   Mean: {np.mean(accs):.4f} | Std: {np.std(accs):.4f}")
    print(f"Recall     Mean: {np.mean(recalls):.4f} | Std: {np.std(recalls):.4f}")
    print(f"F1 Score   Mean: {np.mean(f1s):.4f} | Std: {np.std(f1s):.4f}")
    print(f"AUC        Mean: {np.mean(aucs):.4f} | Std: {np.std(aucs):.4f}")

    # ============ 测试集评估 ============

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]

    print(f"\n[{name}] 测试集评估 (使用最佳模型):")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall  :", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC     :", roc_auc_score(y_test, y_prob))

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc='lower right')
    plt.show()

# ===== 主流程 =====
def run_dual_esm2_transformer(feature_dir1, feature_dir2, label_xlsx):
    # 加载标签
    label_df = pd.read_excel(label_xlsx)
    label_dict = dict(zip(label_df['id'], label_df['label']))

    # 加载特征
    sequences, labels = load_dual_esm2_features(feature_dir1, feature_dir2, label_dict)

    # 初始化 Transformer 编码器
    input_dim = sequences[0].shape[1]
    model = TransformerEncoder(input_dim=input_dim).eval()

    # 提取全局特征
    all_encoded = []
    for seq in sequences:
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float).unsqueeze(0)  # [1, seq_len, dim]
            encoded = model(x)  # [1, d_model]
            all_encoded.append(encoded.squeeze(0).numpy())

    final_features = np.array(all_encoded)

    # 模型评估
    evaluate_model(final_features, labels, name="两种ESM2特征 + Transformer")

# ===== 执行入口 =====
if __name__ == "__main__":
    run_dual_esm2_transformer(
        feature_dir1=r"D:\bishedata2\esm2_t6_features",
        feature_dir2=r"D:\bishedata2\esm2_t12_features",
        label_xlsx=r"D:\bishedata2\铜绿假单细胞+阴性样本 标签.xlsx"
    )
