import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, make_scorer
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

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

# ====== 3. 标准化 ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== 4. 计算类别权重 ======
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
weights_dict = dict(zip(classes, weights))
print(f"类别权重: {weights_dict}")

# ====== 5. 定义初始模型与参数网格 ======
base_model = CatBoostClassifier(
    class_weights=weights,
    verbose=0,
    random_seed=42
)

param_grid = {
    'depth': [4, 6, 8],
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1]
}

# ====== 6. 网格搜索（使用加权 F1 作为评分） ======
scorer = make_scorer(f1_score, average='weighted')
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring=scorer,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

# ====== 7. 划分训练和测试集 ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ====== 8. 执行网格搜索 ======
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"\n最佳参数: {grid_search.best_params_}")

# ====== 9. 自定义交叉验证评估函数 ======
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

# ====== 10. 训练集交叉验证评估 ======
acc_mean, acc_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std = \
    cross_validate_metrics(best_model, X_train, y_train)

# ====== 11. 测试集评估 ======
best_model.fit(X_train, y_train)
test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_auc = roc_auc_score(y_test, test_probs)

# ====== 12. 打印评估结果 ======
print(f"\n========== 合并特征文件：{', '.join(feature_files)} ==========")
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
