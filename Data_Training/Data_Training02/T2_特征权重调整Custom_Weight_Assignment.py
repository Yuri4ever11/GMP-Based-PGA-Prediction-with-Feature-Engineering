import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

# Read data
data = pd.read_csv('DataD.csv')


# Define predefined weight combinations
weight_combinations = [
    [0.015556, 0.015556, 0.015556, 0.015556, 0.015556, 0.015556, 0.015556, 0.015556, 0.015556],
    [0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000],
    [0.004444, 0.004444, 0.004444, 0.004444, 0.004444, 0.004444, 0.004444, 0.004444, 0.004444],
]


# Initialize results DataFrame
results_df = pd.DataFrame(columns=['Model', 'R2 Score', 'Max Error', 'Weights'])

# Iterate through each predefined weight combination
for weights in weight_combinations:
    # Extract features and target variable
    X = data[['Longitude', 'Latitude', 'Magnitude', 'Depth', 'DistanceToDam', 'Azimuth', 'StationLongitude', 'StationLatitude', 'StationElevation']]
    y = data['PeakAcceleration']

    # Data standardization
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply predefined weights
    X_weighted = X * weights

    # Normalize weights
    weights_normalized = normalize([weights], norm='l1').ravel()
    X_weighted_normalized = X_weighted * weights_normalized

    # Normalize data
    X_weighted_normalized = scaler.fit_transform(X_weighted_normalized)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_weighted_normalized, y, test_size=0.2, random_state=42)

    # Impute NaN values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Initialize model
    model = LinearRegression()

    # Cross-validation and prediction
    y_pred = cross_val_predict(model, X_train_imputed, y_train, cv=5)

    # Calculate R2 score and max error
    r2 = r2_score(y_train, y_pred)
    max_error = mean_absolute_error(y_train, y_pred)

    # Save results to DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([[str(model), r2, max_error, weights]],
                                                     columns=['Model', 'R2 Score', 'Max Error', 'Weights'])])

# Save results to CSV file
results_df.to_csv('regression_results_predefined_weights.csv', index=False)
