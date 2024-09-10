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
from sklearn.metrics import root_mean_squared_error
import pickle
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造文件的相对路径（csv文件与代码在同一文件夹内）设置训练集和测试集
train_file_path = os.path.join(current_dir, 'Train.csv')
test_file_path = os.path.join(current_dir, 'Test.csv')

# 读取训练数据和测试数据（注意自己csv文件的字符编码-即excel转csv时你选的编码格式 常见的还有GBK等）
df_train = pd.read_csv(train_file_path, encoding='utf-8')
df_test = pd.read_csv(test_file_path, encoding='utf-8')

# 处理缺失值，使用均值填充-数据用于训练前一定要清洗好
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

# 定义要训练的特征列表，通常而言在回归预测时会有多个特征，为了测试不同特征与目标值的关系，或者能够让预测值与目标值最接近的最佳组合
"""
定义要训练的特征列表，通常而言在回归预测时会有多个特征。
为了测试不同特征与目标值的关系，并找出能够让预测值与目标值最接近的最佳特征组合，需要进行特征选择。
"""
custom_features = [
    # ['Feat1'],
    # ['Feat2'],
    # ['Feat3'],
    # ['Feat4'],
    # ['Feat1', 'Feat2'],
    # ['Feat1', 'Feat3'],
    # ['Feat1', 'Feat4'],
    # ['Feat2', 'Feat3'],
    # ['Feat2', 'Feat4'],
    # ['Feat3', 'Feat4'],
    # ['Feat1', 'Feat2', 'Feat3'],
    # ['Feat1', 'Feat2', 'Feat4'],
    # ['Feat1', 'Feat3', 'Feat4'],
    # ['Feat2', 'Feat3', 'Feat4'],
    ['Feat1', 'Feat2', 'Feat3', 'Feat4', 'Feat5'],

]

# 创建结果DataFrame
results_columns = ['FeatComb', 'Model', 'Train_MSE', 'Train_RMSE', 'Train_MAE', 'Train_R2',
                   'Test_MSE', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Explained_Variance', 'Max_Error']
results_df = pd.DataFrame(columns=results_columns)
# 创建 'models' 文件夹
models_dir = os.path.join(current_dir, 'models')  # 将模型文件夹定义移到循环外部
os.makedirs(models_dir, exist_ok=True)
# 创建预测结果的DataFrame
predictions_df = pd.DataFrame()

for feature in custom_features:
    # 提取特征和目标变量,即你需要的预测特征
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
        'LinearRegressionModel': LinearRegression(),
        'RidgeModel': Ridge(),
        'LassoModel': Lasso(),
        'LarsModel': Lars(),
        'OrthogonalMatchingPursuitModel': OrthogonalMatchingPursuit(),
        'ElasticNetModel': ElasticNet(),
        'DecisionTreeModel': DecisionTreeRegressor(),
        'RandomForestModel': RandomForestRegressor(),
        'GradientBoostingModel': GradientBoostingRegressor(),
        'ExtraTreesModel': ExtraTreesRegressor(),
        'XGBModel': XGBRegressor(),
        # 'LGBMModel': LGBMRegressor(),
        'VotingRegressorModel': VotingRegressor(
            estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor()),
                        ('gb', GradientBoostingRegressor())]),
        'GaussianProcessRBFModel': GaussianProcessRegressor(kernel=None),
        'KNeighborsRegressorModel': KNeighborsRegressor(),
        'PolynomialRegressionModel': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        # 'NeuralNetworkRegressorModel': MLPRegressor(),
        'SGDRegressorModel': SGDRegressor(),
        'PoissonRegressorModel': PoissonRegressor(),
        'PassiveAggressiveRegressorModel': PassiveAggressiveRegressor(),
        'HuberRegressorModel': HuberRegressor(),
        'RidgeCVModel': RidgeCV(),
        'NuSVRModel': NuSVR(),
        'KernelRidgeModel': KernelRidge(),
    }

    # 定义超参数的范围
    """
    超参数对模型的影响非常大，它可以决定模型的性能、复杂度、泛化能力等。
    一个好的超参数设置可以显著提高模型的预测精度，而一个糟糕的超参数设置则可能导致模型过拟合、欠拟合或性能下降。
    【建议确定一个特征组合后，然后从一个大范围的超参数开始调整起，逐渐减小，决定系数大约能有15%的提升或者更多。】
    """
    param_grid = {
        'LinearRegressionModel': {},
        'RidgeModel': {'alpha': [0.1, 1.0, 10.0]},
        'LassoModel': {'alpha': [0.1, 1.0, 10.0]},
        'LarsModel': {},
        'OrthogonalMatchingPursuitModel': {},
        'ElasticNetModel': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.25, 0.5, 0.75]},
        'DecisionTreeModel': {'max_depth': [None, 10, 20, 30]},
        'RandomForestModel': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]},
        'GradientBoostingModel': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
        'ExtraTreesModel': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]},
        'XGBModel': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
        'LGBMModel': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
        'VotingRegressorModel': {},
        'GaussianProcessRBFModel': {},
        'KNeighborsRegressorModel': {'n_neighbors': [5, 10, 15]},
        'polynomialfeatures__degree': [2, 3, 4],
        'linearregression__normalize': [True, False],
        'NeuralNetworkRegressorModel': {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)]},
        'SGDRegressorModel': {'alpha': [0.0001, 0.001, 0.01], 'penalty': ['l1', 'l2']},
        'PoissonRegressorModel': {},
        'PassiveAggressiveRegressorModel': {'C': [0.1, 1.0, 10.0], 'epsilon': [0.1, 0.01, 0.001]},
        'HuberRegressorModel': {'epsilon': [1.1, 1.35, 1.5], 'max_iter': [100, 200, 300]},
        'RidgeCVModel': {},
        'NuSVRModel': {'C': [0.1, 1.0, 10.0], 'degree': [2, 3, 4]},
        'KernelRidgeModel': {'alpha': [0.1, 1.0, 10.0], 'degree': [2, 3, 4]},
    }


    # 训练和评估模型
    for model_name, model in models.items():
        print(f'Training {model_name} for {feature}...')
        grid_search = GridSearchCV(model, param_grid.get(model_name, {}), cv=2, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        #！！！cv=2 使用 2 折交叉验证来评估模型的性能。数据集较大或模型复杂可以使用更大值，较小则可以不用cv=1。！！！
        #！！！n_jobs=-1 表示使用所有的CPU进行训练，如果电脑出现卡顿可以设置为4/8等数值来降低CPU占用。！！！

        # 使用 GridSearchCV 训练模型
        grid_search.fit(X_train_scaled, y_train)

        # Predictions on training set
        y_train_pred = grid_search.predict(X_train_scaled)

        # Predictions on test set
        y_test_pred = grid_search.predict(X_test_scaled)

        # Model evaluation on training set
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)  # 使用 root_mean_squared_error
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        explained_variance_train = explained_variance_score(y_train, y_train_pred)
        max_error_train = max_error(y_train, y_train_pred)

        # Model evaluation on test set
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)  # 使用 root_mean_squared_error
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        explained_variance_test = explained_variance_score(y_test, y_test_pred)
        max_error_test = max_error(y_test, y_test_pred)

        # Save results to the DataFrame
        result_row = [feature, model_name, train_mse, train_rmse, train_mae, train_r2,
                      test_mse, test_rmse, test_mae, test_r2, explained_variance_test, max_error_test]
        results_df = pd.concat([results_df, pd.DataFrame([result_row], columns=results_columns)], ignore_index=True)

        # #【 保存预测结果1-此打印是一列真实值一列预测值 方便对比】
        # temp_df = pd.DataFrame({
        #     'True Values': y_test,
        #     f'{model_name} Predictions': y_test_pred
        # })
        # predictions_df = pd.concat([predictions_df, temp_df], axis=1)
        # 【保存预测结果2 -有了此表格文件可以直接在excel上做折线试图，来直观的观察真实值与预测值的差距】
        if predictions_df.shape[1] == 1:  # 只在第一次添加真实值
            predictions_df['True Values'] = y_test
        predictions_df[f'{model_name} Predictions'] = y_test_pred



# Save Model_Performance results to CSV in the current directory
results_df.to_csv('Model_Performance.csv', index=False)
# 保存预测结果到 CSV 文件
predictions_df.to_csv('Model_Predictions.csv', index=False)

# 保存训练好的模型-慎用模型单独进行预测 这里使用的是pth文件保存的，但通常使用.joblib文件
for model_name, model in models.items():
    for feature in custom_features:
        model_path = os.path.join(models_dir, f'{model_name}_{feature}.pth')
        pickle.dump(grid_search.best_estimator_, open(model_path, 'wb'))