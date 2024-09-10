import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# 读取原始数据, encoding='GBK'
data = pd.read_csv('Train.csv', encoding='GBK')

# 创建SimpleImputer对象，将NaN值用均值填充
imputer = SimpleImputer(strategy='mean')

# 使用imputer对象来填充NaN值
data_imputed = imputer.fit_transform(data)

# 进行PCA分析
pca = PCA(n_components=5)
principal_components = pca.fit_transform(data_imputed)  # 得到5个主成分

# 合并主成分和峰值加速度
data_with_pca = np.column_stack((data_imputed, principal_components))

# 将结果转换为DataFrame, 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14'
data_with_pca_df = pd.DataFrame(data_with_pca, columns=['原始特征1', '原始特征2', '原始特征3', '原始特征4', '原始特征5', '原始特征6', '原始特征7', '原始特征8', '原始特征9', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# 保存新的数据文件
data_with_pca_df.to_csv('D2_PCA5.csv', index=False)
