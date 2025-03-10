import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -------------------- 1. 读取数据 --------------------
file_path = r'D:\bishedata\changgancdhit+阴性样本 aac.xlsx'
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

# -------------------- 3. 训练 & 调优随机森林模型 --------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid, cv=10, scoring='accuracy',
                           n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_


# -------------------- 4. 评估模型 --------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Test Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # 计算 AUC
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_prob)
    print(f"Test AUC: {test_auc:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-AMP', 'AMP'], yticklabels=['Non-AMP', 'AMP'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# 调用评估函数
evaluate_model(best_rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# -------------------- 5. 保存模型 --------------------
save_dir = r"/分类模型"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

model_path = os.path.join(save_dir, "antimicrobial_peptide_model.pkl")
scaler_path = os.path.join(save_dir, "scaler.pkl")

joblib.dump(best_rf_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"模型已保存至: {model_path}")
print(f"标准化模型已保存至: {scaler_path}")
