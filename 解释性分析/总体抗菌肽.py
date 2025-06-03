import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import shap
import umap.umap_ as umap
import seaborn as sns
from shap import summary_plot

# ====== Step 1: 加载与合并特征文件 ======
feature_files = ["apaac.xlsx", "paac.xlsx", "qso.xlsx", "aac.xlsx"]
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"

dfs = [pd.read_excel(os.path.join(base_path, f)) for f in feature_files]
merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on=["SampleName", "label"], how="inner")
print("合并后的维度：", merged.shape)

# ====== Step 2: 特征准备 ======
X = merged.drop(["SampleName", "label"], axis=1, errors="ignore").apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())
y = merged["label"].reset_index(drop=True)
if X.shape[1] == 0:
    raise ValueError("特征为空，请检查特征文件内容。")

# ====== Step 3: 特征选择（线性 SVM 选前100个特征）======
linear_svm = SVC(kernel="linear", probability=True, random_state=42)
linear_svm.fit(X, y)
coef_abs = np.abs(linear_svm.coef_[0])
top_k_features = pd.Series(coef_abs, index=X.columns).nlargest(100).index.tolist()
X_selected = X[top_k_features]

# ====== Step 4: 数据集划分与标准化 ======
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====== Step 5: SVM 定义 ======
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)

# ====== Step 6: 交叉验证评估函数 ======
def cross_validate_metrics(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, recalls, f1s, aucs = [], [], [], []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        accs.append(accuracy_score(y_val, preds))
        recalls.append(recall_score(y_val, preds, average='weighted'))
        f1s.append(f1_score(y_val, preds, average='weighted'))
        aucs.append(roc_auc_score(y_val, probs))
    return np.mean(accs), np.std(accs), np.mean(recalls), np.std(recalls), \
           np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs)

# ====== Step 7: 训练集交叉验证评估 ======
acc_mean, acc_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std = \
    cross_validate_metrics(svm_model, X_train_scaled, y_train)

# ====== Step 8: 测试集评估 ======
svm_model.fit(X_train_scaled, y_train)
test_preds = svm_model.predict(X_test_scaled)
test_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_auc = roc_auc_score(y_test, test_probs)

print(f"\n========== 合并特征文件：{', '.join(feature_files)} ==========")
print(f"【训练集交叉验证】\nAccuracy: {acc_mean:.4f} ± {acc_std:.4f}\n"
      f"Recall:   {rec_mean:.4f} ± {rec_std:.4f}\n"
      f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}\n"
      f"AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
print(f"\n【测试集】\nAccuracy: {test_acc:.4f}\n"
      f"Recall:   {test_rec:.4f}\nF1-score: {test_f1:.4f}\nAUC:      {test_auc:.4f}")
print("===================================")

# ====== Step 9: ROC 曲线绘图（交叉验证 + 测试集）======
def get_cv_roc(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    fpr_list, tpr_list = [], []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y.iloc[train_idx])
        probas_ = model.predict_proba(X[val_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y.iloc[val_idx], probas_)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return fpr_list, tpr_list, mean_fpr, mean_tpr, np.mean(aucs), np.std(aucs)

fpr_list, tpr_list, mean_fpr, mean_tpr, auc_train, std_auc_train = get_cv_roc(
    svm_model, X_train_scaled, y_train)

# 训练 ROC
plt.figure(figsize=(8, 6))
for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    plt.plot(fpr, tpr, lw=1, alpha=0.4, label=f'Fold {i+1}')
plt.plot(mean_fpr, mean_tpr, color='blue', lw=2,
         label=f'Mean ROC (AUC = {auc_train:.3f} ± {std_auc_train:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train ROC Curve (10-fold CV)')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# 测试 ROC
fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='red', lw=2,
         label=f'Test ROC (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ====== Step 10: SHAP 可解释性分析（基于 KernelExplainer）======
explainer = shap.KernelExplainer(svm_model.predict_proba, shap.kmeans(X_train_scaled, 10))
shap_values = explainer.shap_values(X_test_scaled, nsamples=100)
shap.summary_plot(shap_values[1], X_test_scaled, feature_names=top_k_features)

# ====== Step 11: UMAP 降维可视化（全数据）======
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_selected)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="Set1", s=50, alpha=0.9)
plt.title("UMAP Projection of Antimicrobial Peptides")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(title="Label")
plt.grid(True)
plt.tight_layout()
plt.show()
