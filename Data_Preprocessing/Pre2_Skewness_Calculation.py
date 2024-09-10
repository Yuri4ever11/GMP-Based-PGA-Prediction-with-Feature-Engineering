import pandas as pd

# 读取数据
data = pd.read_csv('Train.csv')

# 计算每个特征的偏斜程度
skewness = data.skew()

# 输出结果
print(skewness)