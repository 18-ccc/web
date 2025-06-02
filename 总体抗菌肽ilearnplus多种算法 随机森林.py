import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
)
import os

# 要处理的特征文件名称
feature_files = [
    "apaac.xlsx", "qso.xlsx", "cksaap.xlsx", "ctriad.xlsx",
    "paac.xlsx", "pseaac.xlsx", "aac.xlsx", "DDE.xlsx", "DPC.xlsx"
]
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"

# 手动评估交叉验证训练集的多个指标
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

    return (
        np.mean(accs), np.std(accs),
        np.mean(recalls), np.std(recalls),
        np.mean(f1s), np.std(f1s),
        np.mean(aucs), np.std(aucs)
    )

# 主循环
for file_name in feature_files:
    file_path = os.path.join(base_path, file_name)
    data = pd.read_excel(file_path)

    # 准备数据
    X = data.drop(['SampleName', 'label'], axis=1, errors='ignore')
    y = data['label'].reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())

    # 特征选择（前20个重要特征）
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X, y)
    top_k_features = pd.Series(rf.feature_importances_, index=X.columns).nlargest(20).index.tolist()
    X_selected = X[top_k_features]

    # 标准化 + 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 网格搜索参数空间
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 12, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_scaled, y_train)
    best_rf_model = grid_search.best_estimator_

    # === 手动计算训练集交叉验证的多个指标 ===
    acc_mean, acc_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std = \
        cross_validate_metrics(best_rf_model, X_train_scaled, y_train)

    # 测试集评估
    best_rf_model.fit(X_train_scaled, y_train)
    test_preds = best_rf_model.predict(X_test_scaled)
    test_probs = best_rf_model.predict_proba(X_test_scaled)[:, 1]
    test_acc = accuracy_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds, average='weighted')
    test_f1 = f1_score(y_test, test_preds, average='weighted')
    test_auc = roc_auc_score(y_test, test_probs)

    # 输出
    print(f"\n========== {file_name} ==========")
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
