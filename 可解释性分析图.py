import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from umap import UMAP
import shap
import gc

# ========== Matplotlib 字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== GCN模型定义 ==========
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


# ========== 创建边 ==========
def create_fully_connected_edges(num_nodes):
    row, col = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    return torch.tensor([row, col], dtype=torch.long)


# ========== 构建图数据集 ==========
def build_graph_dataset(feature_dir, label_dict):
    gcn_data_list, esm_pool_features, labels = [], [], []

    for seq_id, label in label_dict.items():
        path = os.path.join(feature_dir, f"{seq_id}.npy")
        if not os.path.exists(path):
            continue
        feat = np.load(path)
        x = torch.tensor(feat, dtype=torch.float)
        edge_index = create_fully_connected_edges(x.shape[0])
        data = Data(x=x, edge_index=edge_index, y=label)
        data.batch = torch.zeros(x.shape[0], dtype=torch.long)  # 全部节点属于一个图
        gcn_data_list.append(data)
        esm_pool_features.append(feat.mean(axis=0))
        labels.append(label)

    return gcn_data_list, np.array(esm_pool_features), np.array(labels)


# ========== SHAP分析函数（修复版本） ==========
def plot_shap_analysis(model, X, feature_names, prefix="", save_dir="."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(f"Raw shap_values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"shap_values list length: {len(shap_values)}")
        print(f"shap_values[0] shape: {np.array(shap_values[0]).shape}")
        print(f"shap_values[1] shape: {np.array(shap_values[1]).shape}")
        shap_values = shap_values[1]  # 取正类SHAP值
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # 取正类SHAP值

    shap_values = np.array(shap_values)
    print(f"X shape: {X.shape}, shap_values shape: {shap_values.shape}, feature_names count: {len(feature_names)}")
    assert shap_values.shape[1] == len(feature_names), "SHAP值与特征数不匹配"

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]

    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(top_features, top_values, color='#1f77b4')
    plt.xlabel('平均绝对SHAP值', fontsize=12)
    plt.title(f'{prefix} - Top 15 特征重要性', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_shap_feature_importance.png"), dpi=300)
    plt.close()

    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X,
        feature_names=feature_names
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f'{prefix} - SHAP蜂群图')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_shap_beeswarm.png"), dpi=300)
    plt.close()

    print(f"✅ SHAP分析完成，图像保存于 {save_dir}")

# ========== UMAP可视化 ==========
def plot_umap(X, y, title, save_path):
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    X_umap = reducer.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1],
                c=['red' if label == 1 else 'blue' for label in y],
                alpha=0.7, edgecolors='k', s=50)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ UMAP图已保存：{save_path}")


# ========== 模型评估主函数 ==========
def evaluate_model(combined_features, labels, feature_names):
    save_dir = "D:/bishedata2/拼接图2"
    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(combined_features)
    print(f"X shape: {X.shape}, X sample:\n{X[:2, :5]}")

    print(f"\nSMOTE 前类别分布: {Counter(labels)}")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, labels)
    print(f"SMOTE 后类别分布: {Counter(y_resampled)}")

    plot_umap(X_resampled, y_resampled, "肠杆菌SMOTE后UMAP", os.path.join(save_dir, "肠杆菌_umap.png"))

    rf = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10], "min_samples_split": [2, 5]}
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)
    best_model = grid_search.best_estimator_
    print(f"最佳模型参数: {grid_search.best_params_}")

    # 交叉验证绘制ROC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(8, 6))
    for i, (train_idx, test_idx) in enumerate(cv.split(X_resampled, y_resampled), start=1):
        X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
        y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
        best_model.fit(X_train, y_train)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.6, label=f'第{i}折 (AUC={roc_auc:.2f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--', lw=2,
             label=f'平均 ROC (AUC={mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('5折交叉验证 ROC 曲线')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "肠杆菌交叉验证_ROC曲线.png"), dpi=300)
    plt.close()

    # 测试集评估
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"测试集召回率: {recall_score(y_test, y_pred):.4f}")
    print(f"测试集F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"测试集AUC: {roc_auc_score(y_test, y_prob):.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC 曲线 (AUC = {roc_auc_score(y_test, y_prob):.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('测试集 ROC 曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "肠杆菌_测试集_ROC曲线.png"), dpi=300)
    plt.close()

    # 使用修复后的SHAP分析函数
    plot_shap_analysis(
        best_model,
        X_test,
        feature_names=feature_names,
        prefix="肠杆菌",
        save_dir=save_dir
    )

    # 清理内存
    del X_resampled, y_resampled, best_model, grid_search
    gc.collect()


# ========== 主流程 ==========
if __name__ == "__main__":
    feature_dir = r"D:\bishedata2\肠杆菌_esm2_model2"
    label_path = r"D:\bishedata2\肠杆菌+阴性样本 标签.xlsx"
    label_df = pd.read_excel(label_path, index_col=0)
    label_dict = label_df['label'].to_dict()

    # 读取GCN和ESM2池化特征
    gcn_data_list, esm_pool_features, labels = build_graph_dataset(feature_dir, label_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = gcn_data_list[0].num_node_features
    gcn_model = GCNEncoder(input_dim).to(device)
    # 假设你有训练好的模型权重，加载
    # gcn_model.load_state_dict(torch.load("path_to_trained_gcn_model.pth"))
    gcn_model.eval()

    gcn_features = []
    with torch.no_grad():
        for data in gcn_data_list:
            data = data.to(device)
            out = gcn_model(data.x, data.edge_index, data.batch)
            gcn_features.append(out.cpu().numpy())
    gcn_features = np.vstack(gcn_features)
    print(f"GCN 特征形状: {gcn_features.shape}")

    # 拼接特征
    combined_features = np.hstack([gcn_features, esm_pool_features])
    print(f"拼接后特征形状: {combined_features.shape}")

    # 特征名称（示例）
    feature_names = [f"GCN_{i}" for i in range(gcn_features.shape[1])] + [f"ESM_{i}" for i in range(esm_pool_features.shape[1])]

    # 模型训练与评估
    evaluate_model(combined_features, labels, feature_names)
