import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('D5.csv')

# 对"峰值加速度(gal)"特征进行log10处理
data['PeakAcceleration'] = np.log10(data['PeakAcceleration'])

# 保存处理后的数据为新的CSV文件
data.to_csv('D5Log10.csv', index=False)

# 打印处理后的数据
print(data.head())

# # 将log10处理后的数值进行反log10处理，恢复为原值
# data['峰值加速度(gal)_反log'] = 10 ** data['峰值加速度(gal)_log']
#
# # 打印处理后的数据
# print(data.head())
