import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
import xgboost as xgb
import os
import pytorch_lightning as pl

class XGBoostLightning(nn.Module):
    def __init__(self, n_estimators=100, max_depth=3):
        super(XGBoostLightning, self).__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def forward(self, x):
        return self.model.predict(xgb.DMatrix(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = mean_squared_error(y.cpu().numpy(), pred.cpu().numpy())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造文件的相对路径
train_file_path = os.path.join(current_dir, 'DataCTrain.csv')
test_file_path = os.path.join(current_dir, 'DataCTest.csv')

# 读取训练数据和测试数据
df_train = pd.read_csv(train_file_path, encoding='utf-8')
df_test = pd.read_csv(test_file_path, encoding='utf-8')

# 处理缺失值，使用均值填充
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

# 定义要训练的特征列表
custom_features = [
  ['Longitude', 'Latitude', 'Magnitude', 'DistanceToDam', 'StationLongitude', 'StationLatitude'],
  ['Latitude', 'Magnitude', 'Depth', 'DistanceToDam', 'StationLongitude', 'StationLatitude'],
  ['Longitude', 'Latitude', 'Magnitude', 'Depth', 'DistanceToDam', 'StationLongitude', 'StationLatitude'],
  ['Longitude', 'Latitude', 'Magnitude', 'Depth', 'DistanceToDam', 'StationLongitude', 'StationLatitude', 'StationElevation', 'Azimuth'],
]

# 创建结果DataFrame
results_columns = ['Feature', 'Model', 'Train_MSE', 'Train_RMSE', 'Train_MAE', 'Train_R2',
                   'Test_MSE', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Explained_Variance', 'Max_Error']
results_df = pd.DataFrame(columns=results_columns)

# 定义超参数的范围
param_grid = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [1,3, 5, 7, 9],
}

# 获取模型
model = XGBoostLightning()

for feature in custom_features:
    # 提取特征和目标变量
    X_train = df_train[feature]
    y_train = df_train['PeakAcceleration']
    X_test = df_test[feature]
    y_test = df_test['PeakAcceleration']

    # 创建 StandardScaler 对象
    scaler = StandardScaler()

    # 使用训练集的均值和标准差进行标准化
    X_train_scaled = scaler.fit_transform(X_train)
    # 使用相同的均值和标准差对测试集进行标准化
    X_test_scaled = scaler.transform(X_test)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 使用 PyTorch-Lightning 进行训练和评估
    trainer = pl.Trainer(max_epochs=10, gpus=1)  # 使用 GPU，如果有的话
    trainer.fit(model, train_loader)

    # 在测试集上评估
    y_test_pred = model(X_test_tensor).detach().numpy()

    # Model evaluation on test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    explained_variance_test = explained_variance_score(y_test, y_test_pred)
    max_error_test = max_error(y_test, y_test_pred)

    # Save results to the DataFrame
    result_row = [feature, 'XGBModel', 0, 0, 0, 0,
                  test_mse, test_rmse, test_mae, test_r2, explained_variance_test, max_error_test]
    results_df = pd.concat([results_df, pd.DataFrame([result_row], columns=results_columns)], ignore_index=True)

# Save results to CSV in the current directory
results_df.to_csv('DataCOutPuts.csv', index=False)
