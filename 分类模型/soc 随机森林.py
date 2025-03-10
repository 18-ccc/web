import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # 引入随机森林
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 加载新的 ASDC 分析结果文件
file_path = r'D:\bishedate\changgancdhit+阴性样本 soc.xlsx'  # 修改为新的文件路径
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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 随机森林 超参数调优
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [1, 2, 4]  # 叶子节点最小样本数
}

# 使用网格搜索和交叉验证
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最优参数训练模型
best_rf_model = grid_search.best_estimator_

# 预测并评估模型
y_pred = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
