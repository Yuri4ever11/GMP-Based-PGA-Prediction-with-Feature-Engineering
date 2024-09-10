import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
fm = FontManager()
for font in fm.ttflist:
    print(font.name)
plt.rcParams['font.sans-serif'] = ['SimHei']

from matplotlib.font_manager import FontProperties
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import pearsonr, kendalltau, spearmanr

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# 读取CSV文件
df = pd.read_csv('E:\earthquake_data\地震预测\step2\Data/D5new.csv')

# 假设data是你的数据框
features = ['Long', 'Lat', 'Mag(M)', 'Depth(km)', 'StationHeight(m)', 'StationLat', 'StationLong', 'DistanceToDam(km)', 'Azimuth']
target = 'MaxAcc(gal)'

# 选择感兴趣的特征列
selected_data = df[features + [target]]

# 计算相关系数矩阵
heatmap_data = selected_data.corr()

# 设置热力图的样式
sns.set(style="white")
fig, ax = plt.subplots(figsize=(15, 10))
cmap = sns.color_palette("Oranges", as_cmap=True)

# 绘制热力图 cmap=cmap
# heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, ax=ax)


# 设置标题
ax.set_title("相关系数热力图", fontproperties='SimHei')

# 设置标签为中文字符
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontproperties='SimHei', rotation=30)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontproperties='SimHei', rotation=0)

# 显示热力图
plt.savefig(' 1热力图')
plt.show()

# 读取数据
#df = pd.read_csv('D2.csv')

# 提取特征和目标列
features = ['Long', 'Lat', 'Mag(M)', 'Depth(km)', 'StationHeight(m)', 'StationLat', 'StationLong', 'DistanceToDam(km)', 'Azimuth']
target = 'MaxAcc(gal)'

# 皮尔逊相关系数  衡量两个变量之间的线性关系  1表示完全正相关，-1表示完全负相关，0表示没有线性关系。
plt.figure(figsize=(15, 12))
correlation_matrix = df[features + [target]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, annot_kws={"size": 15})
plt.title('Pearson Correlation Matrix')
plt.savefig(' 2皮尔逊相关系数.png')
plt.show()

# Spearman Rank 相关系数  不考虑变量的实际值，而是考虑它们的排名（顺序）。该系数用于度量两个变量的单调关系，即如果一个变量增加，另一个变量是否单调地增加或减少。
plt.figure(figsize=(15, 12))
spearman_rank = df[features + [target]].corr(method='spearman')
sns.heatmap(spearman_rank, annot=True, cmap='coolwarm', linewidths=.5, annot_kws={"size": 15})
plt.title('Spearman Rank Correlation Matrix')
plt.savefig(' 3斯皮尔曼等级相关系数.png')
plt.show()

# Kendall Tau 相关系数
plt.figure(figsize=(15, 12))
kendall_tau = df[features + [target]].corr(method='kendall')
sns.heatmap(kendall_tau, annot=True, cmap='coolwarm', linewidths=.5, annot_kws={"size": 15})
plt.title('Kendall Tau Correlation Matrix')
plt.savefig(' 4Kendallta相关系数.png')
plt.show()

# 核密度估计图 估计变量的概率密度函数 发现数据集的概率密度分布情况。
plt.figure(figsize=(15, 12))
sns.kdeplot(df[target], fill=True)
plt.savefig(' 5核密度估计图.png')
plt.show()

# 直方图 显示数据集的分布情况，将数据范围划分为若干个连续的区间（称为“箱”）观察数据的集中趋势和分散程度。
plt.figure(figsize=(15, 12))
sns.histplot(df[target], kde=True)
plt.savefig(' 6直方图.png')
plt.show()

# 箱线图  箱线图显示了数据的中位数、上下四分位数和可能的异常值  检测数据的中心位置、散布和异常值。
plt.figure(figsize=(15, 12))
sns.boxplot(data=df[features + [target]])
plt.savefig(' 7箱线图.png')
plt.show()


# 成对关系图 成对关系图展示了数据集中两两特征之间的关系
plt.figure(figsize=(15, 12))
sns.pairplot(df[features + [target]])
plt.savefig(' 8成对关系图.png')
plt.show()

# 雷达图  雷达图用于显示多个维度的数据。每个维度用一个轴表示，多边形的顶点连接在一起形成图形，展示不同维度的相对大小。
sns.set(style="whitegrid")
radar_df = df[features + [target]].melt(id_vars=target)
plt.figure(figsize=(15, 12))
sns.lineplot(x="variable", y="value", hue="MaxAcc(gal)", data=radar_df, marker="o")
plt.title('Radar Plot')
plt.savefig(' 9雷达图 .png')
plt.show()


# 读取CSV文件
#data = pd.read_csv('D3.csv')

# 特征和目标变量    'StationHeight(m)',
features = ['Long', 'Lat', 'Mag(M)', 'Depth(km)',                   'StationLat', 'StationLong', 'DistanceToDam(km)', 'Azimuth']
target = 'MaxAcc(gal)'

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# 更换中文字体，这里以微软雅黑为例
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 获取数据中的最小值和最大值
min_value = df[features + [target]].min().min()
max_value =df[features + [target]].max().max()

# 创建3x3的子图网格，增加dpi和fontsize，设置水平和垂直间隔
fig, axs = plt.subplots(3, 3, figsize=(18, 18), dpi=180)

# 将axs展平以便于索引
axs = axs.flatten()

# 绘制每个特征的散点图
for i, feature in enumerate(features):
    ax = axs[i]
    ax.scatter(df[target], df[feature], alpha=0.5)
    ax.set_xlabel(target, fontsize=14)  # 调整横坐标轴标签文字大小
    ax.set_ylabel(feature, fontsize=14)  # 调整纵坐标轴标签文字大小
    ax.tick_params(axis='both', which='major', labelsize=12)  # 调整刻度标签文字大小
    ax.set_ylim(min_value, max_value)


# 调整子图布局
plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.92, bottom=0.08, left=0.10, right=0.95)
# 显示图表
plt.savefig(' 10分布特征——散点图 .png')
plt.show()


# 图2
# 设置中文字体
# 创建图表
plt.figure(figsize=(10, 6))

# 绘制每个特征的散点图
for i, feature in enumerate(features):
    plt.scatter(df[target], df[feature], label=feature)

# 设置横坐标和纵坐标标签
plt.xlabel(target)
plt.ylabel('Feature Values')

# 显示图例
plt.legend()

# 显示图表
plt.savefig(' 11总体分布.png')
plt.show()


# 加载数据集
#df = pd.read_csv('E:\earthquake_data\地震预测\step1/D6output.csv')

# 提取特征和目标变量
X = df[['Long', 'Lat', 'Mag(M)', 'Depth(km)', 'StationHeight(m)', 'StationLat', 'StationLong', 'DistanceToDam(km)', 'Azimuth']]
y = df['MaxAcc(gal)']

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 解释预测结果
shap.summary_plot(shap_values, X_train)

# 绘制每个特征的SHAP值分布图
shap.summary_plot(shap_values, X_train, plot_type='bar')

# 绘制每个特征的SHAP值散点图
shap.summary_plot(shap_values, X_train, plot_type='dot')

# 绘制单个样本的SHAP值强度图
sample_index = 0  # 可以更改为其他索引以查看不同样本的结果
shap.force_plot(explainer.expected_value, shap_values[sample_index], X_train.iloc[sample_index], matplotlib=True)

# 模型预测和评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

plt.show()  # 显示所有图形
