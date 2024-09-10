import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('Train.csv')

# 对数转换（Log Transformation）, 'StationLongitude'
features_to_log = ['Latitude','Longitude','StationLongitude']
df[features_to_log] = np.log10(df[features_to_log])

# 平方根转换（Square Root Transformation）
df['Depth'] = np.sqrt(df['Depth'])
# df['StationLongitude'] = np.sqrt(df['StationLongitude'])
# df['Longitude'] = np.sqrt(df['Longitude'])

# 对数转换（Log Transformation）
df['PeakAcceleration'] = np.log10(df['PeakAcceleration'])
df['Magnitude'] = np.log10(df['Magnitude'])
df['DistanceToDam'] = np.log10(df['DistanceToDam'])


# 指数转换（Exponential Transformation）
# df['Longitude'] = np.exp(df['Longitude'])

# 保存处理后的数据
df.to_csv('D2_processed.csv', index=False)