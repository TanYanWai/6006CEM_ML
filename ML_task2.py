import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/Users/yanwai/Desktop/ML_Assignment/adult-dataset/adult.csv"  # Update this path
data = pd.read_csv(file_path)

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Display dataset info and preview
print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

# Remove duplicates
print(f"\nOriginal dataset shape: {data.shape}")
data = data.drop_duplicates()
print(f"Dataset shape after removing duplicates: {data.shape}")

# Check for missing values
print("\nMissing Values Before Removal:")
print(data.isnull().sum())

# Handle missing values by removing rows with any missing values
data = data.dropna()
print("\nDataset shape after removing missing values:", data.shape)

# Detect and handle outliers using IQR
numeric_cols = data.select_dtypes(include=[np.number]).columns
print("\nNumeric Columns:", numeric_cols)

for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

print("\nDataset shape after removing outliers:", data.shape)

# Exploratory Data Analysis (EDA)
# Visualize target column distribution
if '<=50K' in data.columns:
    print("\nIncome Distribution:")
    print(data['<=50K'].value_counts())
    
    sns.countplot(x='<=50K', data=data, palette='Set2')
    plt.title("Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Count")
    plt.show()
else:
    print("\nError: Target column '<=50K' not found!")

# Data Preprocessing
# Handle categorical data with Label Encoding
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_cols)

# Apply Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features (X) and target (y)
X = data.drop('<=50K', axis=1)
y = data['<=50K']

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Before Tuning: Decision Tree Classifier ---
print("\nEvaluating Decision Tree Classifier (Before Tuning)...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred_dt_before = dt.predict(X_test)

# Evaluate Decision Tree Classifier (Before Tuning)
dt_accuracy_before = accuracy_score(y_test, y_pred_dt_before)
print("\nDecision Tree Classifier Performance (Before Tuning):")
print("Accuracy:", dt_accuracy_before)
print("\nClassification Report (Before Tuning):")
print(classification_report(y_test, y_pred_dt_before))

# --- Before Tuning: Logistic Regression ---
print("\nEvaluating Logistic Regression Model (Before Tuning)...")
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Predictions
y_pred_lr_before = lr.predict(X_test)

# Evaluate Logistic Regression Model (Before Tuning)
lr_accuracy_before = accuracy_score(y_test, y_pred_lr_before)
print("\nLogistic Regression Performance (Before Tuning):")
print("Accuracy:", lr_accuracy_before)
print("\nClassification Report (Before Tuning):")
print(classification_report(y_test, y_pred_lr_before))

# --- Hyperparameter tuning for Decision Tree Classifier ---
dt_param_grid = {
    'max_depth': [5, 10, 15, None],  # Depth of tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'criterion': ['gini', 'entropy'],  # Criterion to measure the quality of a split
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider at each split
}

print("\nTraining Decision Tree Classifier with GridSearchCV...")
dt_grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                              param_grid=dt_param_grid, cv=5, n_jobs=-1, scoring='accuracy')
dt_grid_search.fit(X_train, y_train)

# Best parameters for Decision Tree
print("\nBest parameters for Decision Tree:", dt_grid_search.best_params_)

# Predictions
y_pred_dt = dt_grid_search.predict(X_test)

# Evaluate Decision Tree Classifier (After Tuning)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Classifier Performance (After Tuning):")
print("Accuracy:", dt_accuracy)
print("\nClassification Report (After Tuning):")
print(classification_report(y_test, y_pred_dt))

# --- Hyperparameter tuning for Logistic Regression ---
lr_param_grid = {
    'penalty': ['l2', 'none'],  # Regularization
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solver algorithms
    'max_iter': [100, 200, 500]  # Number of iterations for convergence
}

print("\nTraining Logistic Regression Model with GridSearchCV...")
lr_grid_search = GridSearchCV(estimator=LogisticRegression(random_state=42),
                              param_grid=lr_param_grid, cv=5, n_jobs=-1, scoring='accuracy')
lr_grid_search.fit(X_train, y_train)

# Best parameters for Logistic Regression
print("\nBest parameters for Logistic Regression:", lr_grid_search.best_params_)

# Predictions
y_pred_lr = lr_grid_search.predict(X_test)

# Evaluate Logistic Regression Model (After Tuning)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("\nLogistic Regression Performance (After Tuning):")
print("Accuracy:", lr_accuracy)
print("\nClassification Report (After Tuning):")
print(classification_report(y_test, y_pred_lr))

