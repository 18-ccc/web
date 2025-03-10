import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = r'D:\bishedata\changgancdhit aac - mic .xlsx'
data = pd.read_excel(file_path)

# 查看数据的前几行
print(data.head())

# 假设 'SampleName' 列是样本名称，'mic_value' 是目标列
X = data.drop(['SampleName', 'mic_value'], axis=1, errors='ignore')  # 删除非数值列
y = data['mic_value']  # 目标变量（MIC值）

# 数据预处理
# 确保只有数值型数据
X = X.apply(pd.to_numeric, errors='coerce')  # 将非数值型数据转为NaN
X = X.fillna(X.mean())  # 使用均值填充缺失值

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 特征降维（PCA）
# 使用 PCA 降维，保留95%的方差
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 打印降维后的特征数
print(f"Original number of features: {X_train_scaled.shape[1]}")
print(f"Number of features after PCA: {X_train_pca.shape[1]}")

# 选择回归模型
model = RandomForestRegressor(random_state=42, n_jobs=-1)

# 网格搜索超参数
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 内部分裂所需的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点的最小样本数
    'max_features': ['auto', 'sqrt', 'log2']  # 每棵树的最大特征数
}

# 使用网格搜索进行交叉验证
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# 执行网格搜索拟合
grid_search.fit(X_train_pca, y_train)

# 输出最优参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最优参数训练模型
best_model = grid_search.best_estimator_

# 预测
y_train_pred = best_model.predict(X_train_pca)
y_test_pred = best_model.predict(X_test_pca)

# 评估模型
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 输出模型评估结果
print(f"Training MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Training R^2: {train_r2:.4f}")
print(f"Test R^2: {test_r2:.4f}")

# 绘制实际值与预测值对比（测试集）
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual MIC values')
plt.ylabel('Predicted MIC values')
plt.title('Actual vs Predicted MIC values (Test Set)')
plt.show()

# 绘制残差图（测试集）
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.title('Residuals Distribution (Test Set)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
