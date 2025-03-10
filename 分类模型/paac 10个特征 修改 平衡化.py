from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 加载新的 AAC 分析结果文件
file_path = r'D:\bishedata\changgancdhit+阴性样本 apaac.xlsx'
data = pd.read_excel(file_path)

# 查看数据前几行
print(data.head())

# 假设 'SampleName' 是非数值列，'label' 是目标列
X = data.drop(['SampleName', 'label'], axis=1, errors='ignore')  # 删除非数值型列
y = data['label']  # 目标变量

# 确保只包含数值型数据
X = X.apply(pd.to_numeric, errors='coerce')  # 强制转换为数值型，无法转换的将变为 NaN

# 处理缺失值（使用均值填充）
X = X.fillna(X.mean())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化：先划分数据集后进行标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 随机森林超参数调优
param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # 决策树数量
    'max_depth': [10, 20, 30, 40, None],  # 决策树的最大深度
    'min_samples_split': [2, 5, 10, 20],  # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [1, 2, 4, 5, 10],  # 叶子节点最小样本数
    'max_features': ['auto', 'sqrt', 'log2', None]  # 每棵树随机选择的特征数
}

# 使用StratifiedKFold进行交叉验证来保证每个折叠中正负样本的比例保持一致
skf = StratifiedKFold(n_splits=5)

# 使用带有 class_weight='balanced' 的 RandomForest 进行网格搜索
grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced'),
                           param_grid=param_grid,
                           cv=skf,  # 使用StratifiedKFold进行交叉验证
                           scoring='f1',  # 使用F1得分作为评估指标
                           n_jobs=-1,  # 使用所有可用的CPU核心
                           verbose=2)  # 输出详细过程

# 进行网格搜索拟合
grid_search.fit(X_train_scaled, y_train)

# 输出最优参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最优参数训练模型
best_rf_model = grid_search.best_estimator_

# 训练集预测与评估
y_train_pred = best_rf_model.predict(X_train_scaled)
y_train_prob = best_rf_model.predict_proba(X_train_scaled)[:, 1]  # 获取训练集的概率

# 测试集预测与评估
y_test_pred = best_rf_model.predict(X_test_scaled)
y_test_prob = best_rf_model.predict_proba(X_test_scaled)[:, 1]  # 获取测试集的概率

# 训练集准确率与AUC
train_accuracy = best_rf_model.score(X_train_scaled, y_train)
train_auc = roc_auc_score(y_train, y_train_prob)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training AUC: {train_auc:.4f}")

# 测试集准确率与AUC
test_accuracy = best_rf_model.score(X_test_scaled, y_test)
test_auc = roc_auc_score(y_test, y_test_prob)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# 计算并输出 F1 score 和 Recall
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)

print(f"Training F1 Score: {train_f1:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# 混淆矩阵可视化（测试集）
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 绘制ROC曲线（测试集）
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 绘制AUC图（显示AUC的数值）
plt.figure(figsize=(6, 4))
plt.barh(['AUC'], [test_auc], color='skyblue')
plt.xlim(0, 1)
plt.xlabel('AUC Score')
plt.title(f'Test AUC Score = {test_auc:.4f}')
plt.show()

# 输出分类报告
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))
