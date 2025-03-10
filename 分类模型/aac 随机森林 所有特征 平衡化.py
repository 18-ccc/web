import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加载新的 AAC 分析结果文件
file_path = r'D:\bishedata\changgancdhit+阴性样本 aac.xlsx'
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

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 随机森林 超参数调优
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树数量
    'max_depth': [5, 10, 15, 20],  # 减少树的最大深度，避免过拟合
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [1, 2, 4]  # 叶子节点最小样本数
}

# 使用网格搜索和交叉验证
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_scaled, y)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最优参数训练模型
best_rf_model = grid_search.best_estimator_

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 训练集预测与评估
y_train_pred = best_rf_model.predict(X_train)
y_train_prob = best_rf_model.predict_proba(X_train)[:, 1]  # 获取训练集的概率

# 测试集预测与评估
y_test_pred = best_rf_model.predict(X_test)
y_test_prob = best_rf_model.predict_proba(X_test)[:, 1]  # 获取测试集的概率

# 训练集准确率与AUC
train_accuracy = best_rf_model.score(X_train, y_train)
train_auc = roc_auc_score(y_train, y_train_prob)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training AUC: {train_auc:.4f}")

# 测试集准确率与AUC
test_accuracy = best_rf_model.score(X_test, y_test)
test_auc = roc_auc_score(y_test, y_test_prob)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# 输出训练集的分类报告（包含准确率，召回率，F1-score）
print("Training Classification Report:")
train_report = classification_report(y_train, y_train_pred, output_dict=True)
print(classification_report(y_train, y_train_pred))

# 输出测试集的分类报告（包含准确率，召回率，F1-score）
print("Test Classification Report:")
test_report = classification_report(y_test, y_test_pred, output_dict=True)
print(classification_report(y_test, y_test_pred))

# 提取训练集和测试集的召回率和F1-score
train_recall = train_report['1']['recall']  # 正类召回率
train_f1 = train_report['1']['f1-score']  # 正类F1-score
test_recall = test_report['1']['recall']  # 正类召回率
test_f1 = test_report['1']['f1-score']  # 正类F1-score

print(f"Training Recall: {train_recall:.4f}")
print(f"Training F1-score: {train_f1:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-score: {test_f1:.4f}")

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
