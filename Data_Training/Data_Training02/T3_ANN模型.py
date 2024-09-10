import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

# 自定义神经网络模型
class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = self.output_layer(x)
        return x

# 读取数据
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

# 分割特征和目标
X_train = train_data.drop('PeakAcceleration', axis=1)
y_train = train_data['PeakAcceleration']
X_test = test_data.drop('PeakAcceleration', axis=1)
y_test = test_data['PeakAcceleration']

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建模型实例
input_size = X_train.shape[1]
model = CustomModel(input_size)

# 训练模型
num_epochs = 1300

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 定义学习率调度器
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)  # 根据当前的损失值来调整学习率
# 在每个训练周期结束后计算并打印训练集上的 R2 分数和最大误差
    with torch.no_grad():
        model.eval()
        y_train_pred_tensor = model(X_train_tensor)
        model.train()
    # 将 PyTorch Tensor 转换为 NumPy 数组
    y_train_pred = y_train_pred_tensor.numpy()
    # 计算 R2 分数和最大误差
    train_r2 = r2_score(y_train, y_train_pred)
    train_max_error = mean_absolute_error(y_train, y_train_pred)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train R2 Score: {train_r2}, Train Max Error: {train_max_error}')



# 打印每个特征的权重
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f'{name}: {param.data.numpy()}')



# 测试模型
with torch.no_grad():
    model.eval()
    y_pred_tensor = model(X_test_tensor)
    model.train()

# 将 PyTorch Tensor 转换为 NumPy 数组
y_pred = y_pred_tensor.numpy()

# 评估模型性能
r2 = r2_score(y_test, y_pred)
max_error = mean_absolute_error(y_test, y_pred)

print(f'Test R2 Score: {r2}')
print(f'Test Max Error: {max_error}')

"""
Test R2 Score: 0.6878312782903888
Test Max Error: 0.3929892520953723
"""