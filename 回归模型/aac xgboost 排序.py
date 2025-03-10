import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel(
    r"D:\bishedata\changgancdhit+阴性样本 aac - mic - 副本.xlsx",
    engine="openpyxl"
)

# 2. 处理缺失值（如有）
data = data.dropna()  # 或者使用data.fillna()填充

# 3. 特征选择（假设特征在前20列，MIC值在最后一列）
X = data.iloc[:, :20]  # 前20列作为特征
y = data.iloc[:, -1]   # MIC值作为标签

# 4. 特征标准化（使用StandardScaler进行标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 根据MIC值排序并生成排名（排名1为最大MIC值）
data['Rank'] = y.rank(ascending=False)  # 排序后的排名（MIC越大，排名越小）

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test, rank_train, rank_test = train_test_split(
    X_scaled, y, data['Rank'], test_size=0.2, random_state=42
)

# 7. 使用GridSearchCV优化XGBoost的超参数
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_model = XGBRegressor(random_state=42)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, rank_train)

# Get the best model
best_xgb_model = grid_search.best_estimator_

# 8. 在测试集上进行预测
rank_pred = best_xgb_model.predict(X_test)

# 9. 评估模型
mae = mean_absolute_error(rank_test, rank_pred)
r2 = r2_score(rank_test, rank_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"R²: {r2:.3f}")

# 10. 可视化：实际排名 vs 预测排名
plt.scatter(rank_test, rank_pred, alpha=0.6)
plt.plot([rank_test.min(), rank_test.max()], [rank_test.min(), rank_test.max()], "--k")
plt.xlabel("Actual Rank")
plt.ylabel("Predicted Rank")
plt.title("XGBoost - Actual vs Predicted Rank")
plt.show()

# Optional: Cross-validation
cv_scores = cross_val_score(best_xgb_model, X_scaled, rank_train, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-Validation MAE: {np.mean(cv_scores):.3f}")
