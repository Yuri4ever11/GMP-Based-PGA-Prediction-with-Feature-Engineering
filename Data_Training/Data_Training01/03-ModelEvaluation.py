import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
from sklearn.metrics import root_mean_squared_error

"""
PS:该加载模型有时会出现效果不佳的情况，谨慎使用。
"""

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造文件的相对路径（csv文件与代码在同一文件夹内）
test_file_path = os.path.join(current_dir, 'Test.csv')

# 读取测试数据（注意自己csv文件的字符编码）
df_test = pd.read_csv(test_file_path, encoding='utf-8')

# 处理缺失值，使用均值填充
df_test.fillna(df_test.mean(), inplace=True)

# 定义要训练的特征列表
feature = ['Feat1', 'Feat2', 'Feat3', 'Feat4', 'Feat5']  # 替换成你实际使用的特征列表

# 定义要加载的模型名称
model_name = 'LinearRegressionModel'  # 替换成你实际要加载的模型名称

# 创建 'models' 文件夹
models_dir = os.path.join(current_dir, 'models')

# 加载模型
model_path = os.path.join(models_dir, f'{model_name}_{feature}.pth')
model = pickle.load(open(model_path, 'rb'))

# 提取特征
X_test = df_test[feature]

# 创建 StandardScaler 对象
scaler = StandardScaler()

# 使用训练集的均值和标准差进行标准化
X_test_scaled = scaler.fit_transform(X_test)

# 使用模型进行预测
predictions = model.predict(X_test_scaled)

# 获取真实值
y_test = df_test['PeakAcceleration']

# 计算评估指标
test_mse = mean_squared_error(y_test, predictions)
test_rmse = root_mean_squared_error(y_test, predictions)
test_mae = mean_absolute_error(y_test, predictions)
test_r2 = r2_score(y_test, predictions)
explained_variance_test = explained_variance_score(y_test, predictions)
max_error_test = max_error(y_test, predictions)

# 创建结果DataFrame
results_columns = ['Test_MSE', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Explained_Variance', 'Max_Error']
results_df = pd.DataFrame([[test_mse, test_rmse, test_mae, test_r2, explained_variance_test, max_error_test]], columns=results_columns)

# 打印结果DataFrame
print(results_df)

# 保存预测结果到新的DataFrame
predictions_df = pd.DataFrame({'Predicted_PeakAcceleration': predictions})

# 保存预测结果到新的CSV文件
output_file_path = os.path.join(current_dir, 'Predictions.csv')
predictions_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")