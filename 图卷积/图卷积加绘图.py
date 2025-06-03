import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import Counter

def plot_cv_roc_auc(X, y, model=None, n_splits=5):
    print(f"SMOTE 前类别分布: {Counter(y)}")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
    print(f"SMOTE 后类别分布: {Counter(y_resampled)}")

    if model is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, fprs_all, tprs_all = [], [], []

    plt.figure(figsize=(9, 7))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):
        X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
        y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc_val = roc_auc_score(y_val, y_prob)

        aucs.append(auc_val)
        fprs_all.append(fpr)
        tprs_all.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))

        plt.plot(fpr, tpr, lw=2, alpha=0.8, label=f"第{fold + 1}折 AUC={auc_val:.2f}")

    # 平均曲线
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs_all, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='black', linestyle='--',
             label=f"平均 AUC={mean_auc:.2f} ± {std_auc:.2f}", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("假阳性率 (FPR)", fontsize=12)
    plt.ylabel("真正率 (TPR)", fontsize=12)
    plt.title("5折交叉验证 ROC 曲线与 AUC 展示", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
