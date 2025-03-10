from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载新的 AAC 分析结果文件
file_path = r'D:\bishedata\changgancdhit+阴性样本 aac - mic.xlsx'
data = pd.read_excel(file_path)

# 查看数据前几行
print(data.head())

# 假设 'SampleName' 是非数值列，'mic_value' 是目标列（MIC 值）
X = data.drop(['SampleName', 'mic_value'], axis=1, errors='ignore')  # 删除非数值型列
y = data['mic_value']  # 目标变量（MIC值）

# 确保只包含数值型数据
X = X.apply(pd.to_numeric, errors='coerce')  # 强制转换为数值型，无法转换的将变为 NaN

# 处理缺失值（使用均值填充）
X = X.fillna(X.mean())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化：先划分数据集后进行标准化处理
scaler = StandardScaler()

# 训练集标准化
X_train_scaled = scaler.fit_transform(X_train)

# 测试集标准化：使用训练集的均值和方差进行转换
X_test_scaled = scaler.transform(X_test)

# 在进行标准化处理后，恢复列名
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 使用 PCA 来降维，保留95%的方差
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 在 PCA 降维后恢复列名，虽然 PCA 返回的数据没有列名，但我们可以为它们命名为 PC1, PC2, ..., PCn
X_train_pca = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
X_test_pca = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

print(f"Original shape: {X_train.shape}, PCA transformed shape: {X_train_pca.shape}")

# 定义基础回归模型
base_learners = [
    ('xgb', xgb.XGBRegressor(random_state=42, n_jobs=-1)),
    ('lgb', lgb.LGBMRegressor(random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(random_state=42))
]

# 最终回归模型（使用 Ridge 回归）
final_estimator = Ridge()

# 创建 Stacking Regressor
stacking_model = StackingRegressor(estimators=base_learners, final_estimator=final_estimator, n_jobs=-1)

# 定义超参数范围
param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'lgb__n_estimators': [100, 200, 300],
    'lgb__learning_rate': [0.01, 0.05, 0.1],
    'gb__n_estimators': [100, 200, 300],
    'gb__learning_rate': [0.01, 0.05, 0.1]
}

# 使用 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(estimator=stacking_model,
                           param_grid=param_grid,
                           cv=5,  # 5折交叉验证
                           scoring='neg_mean_squared_error',  # 使用负均方误差（MSE）进行评分
                           n_jobs=-1,  # 使用所有可用的CPU核心
                           verbose=2)  # 输出详细过程

# 进行网格搜索拟合
grid_search.fit(X_train_pca, y_train)

# 输出最优参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最优参数训练模型
best_model = grid_search.best_estimator_

# 训练集与测试集的预测
y_train_pred = best_model.predict(X_train_pca)
y_test_pred = best_model.predict(X_test_pca)

# 评估模型性能（计算 MAE、MSE 和 R^2）
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

# 绘制实际值 vs 预测值（测试集）
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