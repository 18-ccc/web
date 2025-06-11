import joblib

model = joblib.load("model1.pkl")
scaler = joblib.load("scaler1.pkl")

print("模型和标准化器加载成功")
