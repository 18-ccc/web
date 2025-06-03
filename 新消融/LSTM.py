import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====== 1. 加载并合并特征文件 ======
feature_files = ["apaac.xlsx", "paac.xlsx", "qso.xlsx", "aac.xlsx"]
base_path = r"D:\bishedata2\总体抗菌肽ilearnplus分析结果"

dfs = []
for file in feature_files:
    path = os.path.join(base_path, file)
    df = pd.read_excel(path)
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on=["SampleName", "label"], how="inner")

print("合并后的维度：", merged.shape)

# ====== 2. 特征与标签准备 ======
X = merged.drop(["SampleName", "label"], axis=1, errors="ignore")
y = merged["label"].reset_index(drop=True)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

if X.shape[1] == 0:
    raise ValueError("特征为空，请检查特征文件内容。")

X_np = X.values  # numpy 格式
y_np = y.values

# ====== 3. PyTorch Dataset ======
class PeptideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== 4. 定义 LSTM 模型 ======
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 5. 训练和评估函数 ======
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = nn.functional.softmax(outputs, dim=1)[:,1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_prob)
    return acc, rec, f1, auc

# ====== 6. 交叉验证 ======
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 32
epochs = 20

accs, recs, f1s, aucs = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np), 1):
    print(f"\nFold {fold}")

    # 分割数据
    X_train_fold, y_train_fold = X_np[train_idx], y_np[train_idx]
    X_val_fold, y_val_fold = X_np[val_idx], y_np[val_idx]

    # 标准化（fit 训练集，transform 验证集）
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)

    # 转换成 LSTM 输入格式
    X_train_fold = X_train_fold.reshape((X_train_fold.shape[0], X_train_fold.shape[1], 1))
    X_val_fold = X_val_fold.reshape((X_val_fold.shape[0], X_val_fold.shape[1], 1))

    # Dataset 和 DataLoader
    train_dataset = PeptideDataset(X_train_fold, y_train_fold)
    val_dataset = PeptideDataset(X_val_fold, y_val_fold)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型和优化器
    model = LSTMClassifier(input_size=1, hidden_size=64, num_layers=2, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # 验证集评估
    acc, rec, f1, auc = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    accs.append(acc)
    recs.append(rec)
    f1s.append(f1)
    aucs.append(auc)

print("\n==== 交叉验证平均结果 ====")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Recall:   {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# ====== 7. 使用全部训练集训练模型，测试集评估 ======
# 划分最终训练测试集
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X_np, y_np, test_size=0.3, random_state=42, stratify=y_np
)

scaler_final = StandardScaler()
X_train_all = scaler_final.fit_transform(X_train_all)
X_test = scaler_final.transform(X_test)

X_train_all = X_train_all.reshape((X_train_all.shape[0], X_train_all.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

train_dataset = PeptideDataset(X_train_all, y_train_all)
test_dataset = PeptideDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model_final = LSTMClassifier(input_size=1, hidden_size=64, num_layers=2, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_final.parameters(), lr=0.001)

print("\n==== 全量训练集训练模型 ====")
for epoch in range(epochs):
    loss = train_one_epoch(model_final, train_loader, criterion, optimizer)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("\n==== 测试集评估 ====")
test_acc, test_rec, test_f1, test_auc = evaluate_model(model_final, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall:   {test_rec:.4f}")
print(f"Test F1-score: {test_f1:.4f}")
print(f"Test AUC:      {test_auc:.4f}")
