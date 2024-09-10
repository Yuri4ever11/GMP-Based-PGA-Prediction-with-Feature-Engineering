import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, OrthogonalMatchingPursuit, ElasticNet, HuberRegressor, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, PoissonRegressor, PassiveAggressiveRegressor
from sklearn.kernel_ridge import KernelRidge
import os

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


    # 获取模型
    models = {
        'XGBModel': XGBRegressor(),
    }

    # 定义超参数的范围
    param_grid = {
        'XGBModel': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
    }

    # 训练和评估模型
    for model_name, model in models.items():
        print(f'Training {model_name} for {feature}...')
        grid_search = GridSearchCV(model, param_grid.get(model_name, {}), cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=12)
        grid_search.fit(X_train_scaled, y_train)

        # Predictions on training set
        y_train_pred = grid_search.predict(X_train_scaled)

        # Predictions on test set
        y_test_pred = grid_search.predict(X_test_scaled)

        # Model evaluation on training set
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        explained_variance_train = explained_variance_score(y_train, y_train_pred)
        max_error_train = max_error(y_train, y_train_pred)

        # Model evaluation on test set
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        explained_variance_test = explained_variance_score(y_test, y_test_pred)
        max_error_test = max_error(y_test, y_test_pred)

        # Save results to the DataFrame
        result_row = [feature, model_name, train_mse, train_rmse, train_mae, train_r2,
                      test_mse, test_rmse, test_mae, test_r2, explained_variance_test, max_error_test]
        results_df = pd.concat([results_df, pd.DataFrame([result_row], columns=results_columns)], ignore_index=True)

# Save results to CSV in the current directory
results_df.to_csv('DataFOutPuts.csv', index=False)
