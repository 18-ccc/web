import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

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


def train_random_forest(features, targets):
    """
    训练随机森林回归模型。

    参数:
        features (DataFrame): 特征数据。
        targets (Series): 目标变量（MIC值）。

    返回:
        model: 训练好的随机森林回归模型。
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

    # 创建随机森林回归模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    return model


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
aac_file = "D:/bishedate/AACchanggan副本.xlsx"  # 包含AAC特征的XLS/XLSX文件
mic_file = "D:/bishedate/肠杆菌mic_values - 副本.xlsx"  # 包含MIC值的XLS/XLSX文件
output_model_path = "D:/bishedate/random_forest_model.pkl"  # 模型保存路径

# 加载数据
features, targets = load_data(aac_file, mic_file)

# 训练随机森林回归模型
model = train_random_forest(features, targets)

# 保存模型
save_model(model, output_model_path)