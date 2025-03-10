import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import xgboost as xgb  # 导入XGBoost库


def load_data(aac_file, mic_file):
    """
    从AAC分析文件和MIC值文件中加载数据。

    参数:
        aac_file (str): 包含氨基酸组成的XLS/XLSX文件路径。
        mic_file (str): 包含MIC值的XLS/XLSX文件路径。

    返回:
        features (DataFrame): AAC特征。
        targets (Series): MIC值。
    """
    # 加载AAC特征
    features = pd.read_excel(aac_file, index_col=0)  # 使用read_excel读取XLS/XLSX文件
    features.index.name = "id"  # 确保ID列作为索引

    # 加载MIC值
    mic_df = pd.read_excel(mic_file)  # 使用read_excel读取XLS/XLSX文件
    # 合并特征和目标变量
    data = features.join(mic_df.set_index("id"), on="id")
    data.dropna(inplace=True)  # 删除缺失值

    # 分离特征和目标变量
    features = data.drop(columns=["mic"])
    targets = data["mic"]

    return features, targets


def train_xgboost(features, targets):
    """
    训练XGBoost回归模型，并使用交叉验证进行参数调优。

    参数:
        features (DataFrame): 特征数据。
        targets (Series): 目标变量（MIC值）。

    返回:
        model: 训练好的XGBoost回归模型。
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

    # 定义XGBoost回归模型
    model = xgb.XGBRegressor(random_state=42)

    # 定义参数网格
    param_grid = {
        "n_estimators": [50, 100, 150],  # 树的数量
        "learning_rate": [0.01, 0.1, 0.2],  # 学习率
        "max_depth": [3, 5, 6, 8],  # 最大树深度
        "subsample": [0.7, 0.8, 0.9],  # 子采样率
        "colsample_bytree": [0.7, 0.8, 0.9]  # 列采样率
    }

    # 使用GridSearchCV进行参数调优
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 评估模型
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return best_model


def save_model(model, output_path):
    """
    保存训练好的模型。

    参数:
        model: 训练好的模型。
        output_path (str): 模型保存路径。
    """
    joblib.dump(model, output_path)
    print(f"模型已保存到：{output_path}")


# 指定文件路径
aac_file = "D:/bishedate/AACchanggancdhit+阴性样本data.xlsx"  # 包含AAC特征的XLS/XLSX文件
mic_file = "D:/bishedate/肠杆菌mic_values.xlsx"  # 包含MIC值的XLS/XLSX文件
output_model_path = "D:/bishedate/xgboost_model.pkl"  # 模型保存路径

# 加载数据
features, targets = load_data(aac_file, mic_file)

# 训练XGBoost回归模型并进行参数调优
model = train_xgboost(features, targets)

# 保存模型
save_model(model, output_model_path)