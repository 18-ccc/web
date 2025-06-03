import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from umap import UMAP
import shap
import matplotlib.pyplot as plt

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ========== 1. 定义 GCN 模型 ==========
class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

# ========== 2. 构造图结构 ==========
def create_fully_connected_edges(num_nodes):
    row, col = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    return torch.tensor([row, col], dtype=torch.long)

# ========== 3. 加载特征并构建图数据 ==========
def build_graph_dataset(feature_dir, label_dict):
    gcn_data_list, pooling_features, labels = [], [], []

    for seq_id, label in label_dict.items():
        path = os.path.join(feature_dir, f"{seq_id}.npy")
        if not os.path.exists(path):
            continue

        feat = np.load(path)
        x = torch.tensor(feat, dtype=torch.float)
        edge_index = create_fully_connected_edges(x.shape[0])
        data = Data(x=x, edge_index=edge_index, y=label)
        data.batch = torch.zeros(x.shape[0], dtype=torch.long)

        gcn_data_list.append(data)
        pooling_features.append(feat.mean(axis=0))
        labels.append(label)

    return gcn_data_list, np.array(pooling_features), np.array(labels)

# ========== 4. UMAP可视化并保存 ==========
def plot_umap(X, y, title="UMAP Visualization"):
    reducer = UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_umap[y==0, 0], X_umap[y==0, 1], label="Negative", alpha=0.6)
    plt.scatter(X_umap[y==1, 0], X_umap[y==1, 1], label="Positive", alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()
    save_path = "SMOTE后特征分布_UMAP.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ UMAP图已保存：{save_path}")
    plt.close()

# ========== 5. SHAP解释并保存 ==========
def plot_shap_summary(model, X, feature_names=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    else:
        shap_vals_to_plot = shap_values

    plt.title("SHAP Summary Plot (Feature Importance)")
    shap.summary_plot(shap_vals_to_plot, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    save_path = "特征重要性_SHAP总结图.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ SHAP总结图已保存：{save_path}")
    plt.close()

# ========== 6. 训练 + 网格搜索评估模型 ==========
def evaluate_model(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"\nSMOTE 前类别分布: {Counter(y)}")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
    print(f"SMOTE 后类别分布: {Counter(y_resampled)}")

    # ———— 保存 UMAP 可视化 ————
    plot_umap(X_resampled, y_resampled, title="SMOTE后特征分布_UMAP")

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

    print("\n=== 拼接特征：交叉验证评估 ===")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Recall  : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"AUC     : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # 测试集评估
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                               cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]

    print("\n=== 拼接特征：测试集评估 ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall  :", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("AUC     :", roc_auc_score(y_test, y_prob))

    # ———— 保存 SHAP解释图 ————
    plot_shap_summary(best_rf, X_test)

# ========== 7. 主流程整合 ==========
def run_combined_feature_model(feature_dir, label_xlsx):
    label_df = pd.read_excel(label_xlsx)
    label_dict = dict(zip(label_df['id'], label_df['label']))

    gcn_data_list, pooling_feats, labels = build_graph_dataset(feature_dir, label_dict)
    input_dim = gcn_data_list[0].x.shape[1]

    # 提取 GCN 特征
    gcn = GCNEncoder(input_dim).eval()
    gcn_features = []
    for data in gcn_data_list:
        with torch.no_grad():
            vec = gcn(data.x, data.edge_index, data.batch)
            gcn_features.append(vec.cpu().numpy().flatten())

    combined = np.concatenate([pooling_feats, np.array(gcn_features)], axis=1)
    evaluate_model(combined, labels)

# ========== 8. 入口 ==========
if __name__ == "__main__":
    run_combined_feature_model(
        feature_dir=r"D:\bishedata\鲍曼不动杆菌_esm2_model2",
        label_xlsx=r"D:\bishedata\鲍曼不动杆菌+阴性样本 标签.xlsx"
    )