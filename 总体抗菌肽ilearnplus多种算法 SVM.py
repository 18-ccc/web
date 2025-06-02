import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import os

# 要合并的特征文件列表
selected_files = ["apaac.xlsx", "paac.xlsx"]
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"

# 合并特征文件
dfs = []
for file in selected_files:
    path = os.path.join(base_path, file)
    df = pd.read_excel(path)
    dfs.append(df)

# 使用 SampleName 做内连接，确保样本顺序对齐
merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on=['SampleName', 'label'], how='inner')

print("合并后的维度：", merged.shape)
print("列名预览：", merged.columns.tolist())

# 准备数据
X = merged.drop(['SampleName', 'label'], axis=1, errors='ignore')
y = merged['label'].reset_index(drop=True)

# 检查是否为空
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())
if X.shape[1] == 0:
    raise ValueError("所有特征列都为空，可能是特征文件格式或列名不一致问题，请检查。")

# 特征选择（选出前30重要特征）
catboost_tmp = CatBoostClassifier(verbose=0, random_state=42)
catboost_tmp.fit(X, y)
importances = pd.Series(catboost_tmp.get_feature_importance(), index=X.columns)
top_features = importances.nlargest(60).index.tolist()
X_selected = X[top_features]

# 数据划分 + 标准化
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 网格搜索
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.03, 0.1],
    'iterations': [200, 500],
}
grid = GridSearchCV(
    CatBoostClassifier(verbose=0, random_state=42),
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_

# 手动交叉验证评估
def evaluate_cross_val(model, X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, recalls, f1s, aucs = [], [], [], []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict(X[val_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]

        accs.append(accuracy_score(y.iloc[val_idx], preds))
        recalls.append(recall_score(y.iloc[val_idx], preds, average='weighted'))
        f1s.append(f1_score(y.iloc[val_idx], preds, average='weighted'))
        aucs.append(roc_auc_score(y.iloc[val_idx], probs))
    return np.mean(accs), np.std(accs), np.mean(recalls), np.std(recalls), np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs)

acc_mean, acc_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std = evaluate_cross_val(best_model, X_train_scaled, y_train)

# 测试集评估
best_model.fit(X_train_scaled, y_train)
test_preds = best_model.predict(X_test_scaled)
test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds, average='weighted')
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_auc = roc_auc_score(y_test, test_probs)

# 输出结果
print(f"\n========== 合并特征分析（CatBoost）==========")
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
print("=========================================")
