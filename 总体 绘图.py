import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
import umap

warnings.filterwarnings("ignore")

# ---------- 设置matplotlib中文黑体字体 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 1. 加载合并特征文件
feature_files = ["apaac.xlsx", "paac.xlsx", "qso.xlsx"]
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"
dfs = [pd.read_excel(os.path.join(base_path, file)) for file in feature_files]

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on=["SampleName", "label"], how="inner")

# 2. 特征与标签准备
X = merged.drop(["SampleName", "label"], axis=1, errors="ignore")
y = merged["label"].reset_index(drop=True)
X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean())

# 3. 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 用线性L1 SVM做特征选择，选前90个特征
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=5000,
                 class_weight='balanced', random_state=42)
lsvc.fit(X_train_scaled, y_train)
coef_abs = np.abs(lsvc.coef_)[0]
top90_idx = np.argsort(coef_abs)[::-1][:90]
top90_features = X.columns[top90_idx]

print("选出的前90个特征:")
print(top90_features.tolist())

# 构造选特征后的训练测试集
X_train_selected = pd.DataFrame(X_train_scaled, columns=X.columns)[top90_features].values
X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[top90_features].values

# 6. UMAP降维可视化（训练集）
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
X_train_umap = reducer.fit_transform(X_train_selected)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_train_umap[:, 0], X_train_umap[:, 1],
    c=y_train,  # 直接使用标签
    cmap='bwr',  # 使用红蓝配色方案
    alpha=0.7, edgecolors='k', s=50
)
plt.xlabel('UMAP1', fontsize=14, fontweight='bold')
plt.ylabel('UMAP2', fontsize=14, fontweight='bold')
plt.title('训练集选特征后的UMAP降维可视化', fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend(*scatter.legend_elements(), labels=['阴性样本', '阳性样本'], title='类别', loc='upper right')
plt.tight_layout()
plt.show()

# 7. 定义SVM模型
svm_model = SVC(
    kernel="rbf", C=1.0, gamma="scale",
    probability=True, class_weight="balanced", random_state=42
)

# 8. 交叉验证及绘制每折ROC曲线，及平均ROC曲线
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

plt.figure(figsize=(10, 8))
accs, recalls, f1s, aucs = [], [], [], []

tprs = []
mean_fpr = np.linspace(0, 1, 100)

for i, (train_idx, valid_idx) in enumerate(skf.split(X_train_selected, y_train)):
    X_tr, X_val = X_train_selected[train_idx], X_train_selected[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    svm_model.fit(X_tr, y_tr)
    preds = svm_model.predict(X_val)
    probs = svm_model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    rec = recall_score(y_val, preds, average='weighted')
    f1 = f1_score(y_val, preds, average='weighted')
    auc_score = roc_auc_score(y_val, probs)

    accs.append(acc)
    recalls.append(rec)
    f1s.append(f1)
    aucs.append(auc_score)

    fpr, tpr, _ = roc_curve(y_val, probs)
    plt.plot(fpr, tpr, lw=1.5, alpha=0.6, label=f'Fold {i+1} (AUC = {auc_score:.3f})')

    # 插值tpr，用于平均曲线
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

# 画平均ROC曲线
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = roc_auc_score(y_train, svm_model.predict_proba(X_train_selected)[:,1])
plt.plot(mean_fpr, mean_tpr, color='b', lw=3, linestyle='--', label=f'平均ROC (AUC = {np.mean(aucs):.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
plt.xlabel('假阳性率', fontsize=14, fontweight='bold', fontname='SimHei')
plt.ylabel('真正率', fontsize=14, fontweight='bold', fontname='SimHei')
plt.title('训练集10折交叉验证 ROC 曲线', fontsize=16, fontweight='bold', fontname='SimHei')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n【训练集10折交叉验证指标均值±标准差】")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Recall:   {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# 9. 测试集评估
svm_model.fit(X_train_selected, y_train)
test_preds = svm_model.predict(X_test_selected)
test_probs = svm_model.predict_proba(X_test_selected)[:, 1]

test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_auc = roc_auc_score(y_test, test_probs)

print("\n【测试集评估】")
print(f"Accuracy: {test_acc:.4f}")
print(f"Recall:   {test_rec:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"AUC:      {test_auc:.4f}")

# 10. 测试集ROC曲线
fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='r', lw=2, label=f'测试集 ROC (AUC={test_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
plt.xlabel('假阳性率', fontsize=14, fontweight='bold', fontname='SimHei')
plt.ylabel('真正率', fontsize=14, fontweight='bold', fontname='SimHei')
plt.title('测试集 ROC 曲线', fontsize=16, fontweight='bold', fontname='SimHei')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
