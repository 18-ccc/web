import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter

# ===== GCN 图卷积模块 =====
class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

# ===== 创建全连接图结构 =====
def create_fully_connected_edges(num_nodes):
    row = []
    col = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index

# ===== 构建图数据列表 =====
def build_graph_dataset(feature_dir, label_dict):
    gcn_data_list = []
    pooling_features = []
    labels = []
    not_found = []

    for seq_id, label in label_dict.items():
        path = os.path.join(feature_dir, f"{seq_id}.npy")
        if not os.path.exists(path):
            not_found.append(seq_id)
            continue
        feat = np.load(path)
        x = torch.tensor(feat, dtype=torch.float)
        edge_index = create_fully_connected_edges(x.shape[0])
        data = Data(x=x, edge_index=edge_index, y=label)
        data.batch = torch.zeros(x.shape[0], dtype=torch.long)
        gcn_data_list.append(data)
        pooling_features.append(feat.mean(axis=0))  # 平均池化特征
        labels.append(label)

    print(f"\n加载序列总数: {len(labels)}, 缺失序列数: {len(not_found)}")
    return gcn_data_list, np.array(pooling_features), np.array(labels)

# ===== 评估函数（含训练集交叉验证） =====
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

    # 网格搜索 RandomForestClassifier 超参数
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

# ===== 主流程 =====
def run_three_feature_comparisons(feature_dir, label_xlsx):
    # === 加载标签 ===
    label_df = pd.read_excel(label_xlsx)
    label_dict = dict(zip(label_df['id'], label_df['label']))

    # === 构建数据集 ===
    gcn_data_list, pooling_feats, labels = build_graph_dataset(feature_dir, label_dict)

    # === 初始化 GCN ===
    input_dim = gcn_data_list[0].x.shape[1]
    gcn = GCNEncoder(input_dim).eval()

    # === 提取 GCN 特征 ===
    gcn_features = []
    for data in gcn_data_list:
        with torch.no_grad():
            vec = gcn(data.x, data.edge_index, data.batch)
            gcn_features.append(vec.cpu().numpy().flatten())
    gcn_features = np.array(gcn_features)

    # === 三种特征输入进行评估 ===
    evaluate_model(pooling_feats, labels, name="仅 ESM2 平均池化特征")
    evaluate_model(gcn_features, labels, name="仅 GCN 图卷积特征")
    combined = np.concatenate([pooling_feats, gcn_features], axis=1)
    evaluate_model(combined, labels, name="拼接 ESM2 + GCN 特征")

# ===== 执行 =====
if __name__ == "__main__":
    run_three_feature_comparisons(
        feature_dir=r"D:\bishedata2\铜绿假单细胞_esm2_model2",
        label_xlsx=r"D:\bishedata2\铜绿假单细胞+阴性样本 标签.xlsx"
    )