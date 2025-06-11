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
import warnings
import joblib

warnings.filterwarnings("ignore")

# ========== 1. 加载 AAC 特征文件 ==========
feature_file = "aac.xlsx"
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"
df = pd.read_excel(os.path.join(base_path, feature_file))

# ========== 2. 特征准备 ==========
X = df.drop(["SampleName", "label"], axis=1, errors="ignore")
y = df["label"].reset_index(drop=True)
X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean())

# ========== 3. 数据划分与标准化 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 定义 SVM 模型 ==========
svm_model = SVC(
    kernel="rbf", C=1.0, gamma="scale",
    probability=True, class_weight="balanced", random_state=42
)

# ========== 5. 交叉验证指标计算 ==========
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

acc_mean, acc_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std = \
    cross_validate_metrics(svm_model, X_train_scaled, y_train)

# ========== 6. 绘制交叉验证10折ROC曲线（每折曲线均显示AUC） ==========
def plot_cv_roc_curve(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train_cv, X_val_cv = X[train_idx], X[valid_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train_cv, y_train_cv)
        probas_ = model.predict_proba(X_val_cv)[:, 1]
        fpr, tpr, _ = roc_curve(y_val_cv, probas_)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        plt.plot(fpr, tpr, lw=1.5, alpha=0.6,
                 label=f'Fold {i + 1} ROC (AUC = {roc_auc:.3f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
             lw=3, alpha=0.9)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    plt.xlabel('假阳性率', fontsize=14)
    plt.ylabel('真正率', fontsize=14)
    plt.title('训练集 10折交叉验证 ROC 曲线', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_cv_roc_curve(svm_model, X_train_scaled, y_train, n_splits=10)

# ========== 7. 测试集评估 ==========
svm_model.fit(X_train_scaled, y_train)
test_preds = svm_model.predict(X_test_scaled)
test_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_auc = roc_auc_score(y_test, test_probs)

print("\n【训练集交叉验证】")
print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"Recall:   {rec_mean:.4f} ± {rec_std:.4f}")
print(f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"AUC:      {auc_mean:.4f} ± {auc_std:.4f}")

print("\n【测试集】")
print(f"Accuracy: {test_acc:.4f}")
print(f"Recall:   {test_rec:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"AUC:      {test_auc:.4f}")

# ========== 8. 测试集 ROC 曲线 ==========
fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='r', lw=2, label=f'测试集 ROC (AUC={roc_auc_test:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)
plt.xlabel('假阳性率')
plt.ylabel('真正率')
plt.title('测试集 ROC 曲线')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 9. 保存 SVM 模型和标准化器 ==========
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
