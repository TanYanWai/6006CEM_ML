import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/Users/yanwai/Desktop/ML_Assignment/covid_data copy 2.csv')

# Explore the data 
print("First few rows of the dataset:")
print(df.head())

print("\nData information:")
print(df.info())

# Check for missing values and print the number of missing values for each column
missing_values = df.isnull().sum()
print("\nNumber of missing values in each column:")
print(missing_values)

# Remove duplicate rows, if any
df = df.drop_duplicates()

# Check if duplicates were removed
print(f"\nNumber of rows after removing duplicates: {df.shape[0]}")

# Filter the necessary columns
df = df[['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'icu_patients', 
          'hosp_patients', 'total_tests', 'new_tests', 'positive_rate', 
          'total_vaccinations', 'people_fully_vaccinated']]

# Drop rows with missing values
df.dropna(inplace=True)

# Remove outliers using Z-score
z_scores = zscore(df)  # Calculate Z-scores for all columns
df = df[(abs(z_scores) < 3).all(axis=1)]  # Keep rows where all Z-scores are < 3

print(f"\nNumber of rows after removing outliers: {df.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
df[['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'icu_patients', 
    'hosp_patients', 'total_tests', 'new_tests', 'positive_rate', 
    'total_vaccinations', 'people_fully_vaccinated']] = scaler.fit_transform(
        df[['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
            'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 
            'positive_rate', 'total_vaccinations', 'people_fully_vaccinated']])

# Set features and target 
X = df[['new_cases', 'total_deaths', 'new_deaths', 'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 
        'positive_rate', 'total_vaccinations', 'people_fully_vaccinated']]  # Features (input variables)
y = df['total_cases']  # Target (what you are predicting)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

param_grid = {
    'n_estimators': [50, 75, 100],  # Fewer trees for faster training
    'max_depth': [5, 10, 15],  # Smaller depths to reduce overfitting and computation
    'min_samples_split': [5, 10],  # Higher split threshold to simplify trees
    'min_samples_leaf': [2, 4],  # Avoid creating leaves with too few samples
    'max_features': ['sqrt', 'log2'],  # Limit features to speed up training
    'bootstrap': [True]  # Only use bootstrap sampling for consistency
}

rf_model = RandomForestRegressor(random_state=42)

# Grid Search with more extensive cross-validation (10 folds instead of 5)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters and model
best_rf_model = grid_search.best_estimator_
print("\nBest parameters from GridSearchCV:")
print(grid_search.best_params_)

# Evaluate Linear Regression Model
print("\nLinear Regression Model Evaluation:")
print("Linear Regression - Test Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression - Test R2 Score:", r2_score(y_test, y_pred_lr))

# Evaluate Random Forest on training data with tuned parameters
y_pred_rf_train = best_rf_model.predict(X_train)
print("\nTuned Random Forest Regressor - Training Mean Squared Error:", mean_squared_error(y_train, y_pred_rf_train))
print("Tuned Random Forest Regressor - Training R2 Score:", r2_score(y_train, y_pred_rf_train))

# Evaluate Random Forest on test data
y_pred_rf = best_rf_model.predict(X_test)
print("\nTuned Random Forest Regressor - Test Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("Tuned Random Forest Regressor - Test R2 Score:", r2_score(y_test, y_pred_rf))

# Cross-validation for Tuned Random Forest
cv_scores = cross_val_score(best_rf_model, X, y, cv=10, scoring='r2')  # Increase cv to 10 folds
print("\nTuned Random Forest Regressor - Cross-Validation R2 Scores:", cv_scores)
print("Tuned Random Forest Regressor - Mean Cross-Validation R2 Score:", cv_scores.mean())

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.xlim(y_test.min(), y_test.max())
    plt.ylim(y_test.min(), y_test.max())
    plt.grid()
    plt.show()

# Plot for Linear Regression
plot_actual_vs_predicted(y_test, y_pred_lr, 'Linear Regression: Actual vs Predicted')

# Plot for Tuned Random Forest Regressor
plot_actual_vs_predicted(y_test, y_pred_rf, 'Tuned Random Forest Regressor: Actual vs Predicted')
