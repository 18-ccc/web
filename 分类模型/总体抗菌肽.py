import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import joblib  # 用于保存和加载模型

# -------------------- 1. 读取数据 --------------------
file_path = r'D:\bishedata\总体抗菌肽 aac.xlsx'
data = pd.read_excel(file_path)

# 假设 'SampleName' 是非数值列，'label' 是目标列
X = data.drop(['SampleName', 'label'], axis=1, errors='ignore')
y = data['label']

# 处理非数值数据
X = X.apply(pd.to_numeric, errors='coerce')  # 确保所有列都是数值
X = X.fillna(X.mean())  # 用均值填充缺失值

# -------------------- 2. 数据划分和标准化 --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- 3. 设置超参数网格 --------------------
param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [10, 12, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# -------------------- 4. 使用 GridSearchCV 进行超参数优化 --------------------
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数和最佳交叉验证分数
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# -------------------- 5. 使用最佳模型进行训练和评估 --------------------
best_rf_model = grid_search.best_estimator_

# 重新训练最佳模型
best_rf_model.fit(X_train_scaled, y_train)

# 在训练集和测试集上的预测
train_predictions = best_rf_model.predict(X_train_scaled)
test_predictions = best_rf_model.predict(X_test_scaled)

# 评估模型
def evaluate_model():
    print(f"Training Accuracy: {best_rf_model.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {best_rf_model.score(X_test_scaled, y_test):.4f}")

    # 计算训练集和测试集的 Precision, Recall, F1-score
    train_precision = precision_score(y_train, train_predictions, average='weighted')
    train_recall = recall_score(y_train, train_predictions, average='weighted')
    train_f1 = f1_score(y_train, train_predictions, average='weighted')

    test_precision = precision_score(y_test, test_predictions, average='weighted')
    test_recall = recall_score(y_test, test_predictions, average='weighted')
    test_f1 = f1_score(y_test, test_predictions, average='weighted')

    print(f"\nTraining Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}")

    # 计算混淆矩阵
    print("\nTraining Confusion Matrix:")
    print(confusion_matrix(y_train, train_predictions))
    print("\nTest Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    # 计算 AUC
    train_auc = roc_auc_score(y_train, best_rf_model.predict_proba(X_train_scaled)[:, 1])
    test_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test_scaled)[:, 1])

    print(f"Training AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

# 调用评估函数
evaluate_model()

# -------------------- 6. 保存模型和标准化器 --------------------
# 保存最佳模型
joblib.dump(best_rf_model, r"D:\bishe\毕业设计\分类模型\antimicrobial_peptide_model.pkl")

# 保存标准化器
joblib.dump(scaler, r"D:\bishe\毕业设计\分类模型\scaler.pkl")
