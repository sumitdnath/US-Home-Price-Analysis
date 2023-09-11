# Ensemble Model with Random Forest Regressor for US Home Price Analysis

## Introduction

In this documentation, we will walk you through the process of building an ensemble model using a Random Forest Regressor with hyperparameter tuning to analyze and predict key factors that influence US home prices on a national scale over the last 20 years. This data science project aims to explain how these factors have impacted home prices.

## Problem Statement

Understanding the factors influencing US home prices is vital for various stakeholders, including homeowners, real estate investors, and policymakers. By developing a data science model, we can gain insights into the relationships between these factors and home prices, enabling better decision-making in the real estate market.

## Implementation Steps

Let's break down the implementation into several steps:

### 1. Data Acquisition

We start by obtaining the dataset named `CSUSHPISA.csv`. This dataset contains historical home price data along with associated dates.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the provided dataset
data = pd.read_csv('CSUSHPISA.csv')
```
### 2. Data Preprocessing
• Convert the 'DATE' column to a datetime format.
• Extract relevant features from the date, such as 'Year,' 'Month,' and 'Quarter.'

```python
# Data preprocessing: Extract features from the date
data['DATE'] = pd.to_datetime(data['DATE'])
data['Year'] = data['DATE'].dt.year
data['Month'] = data['DATE'].dt.month
data['Quarter'] = data['DATE'].dt.quarter
```

### 3. Feature Selection
We choose the 'Year,' 'Month,' and 'Quarter' features to analyze how these time-related factors impact home prices. No additional external data is merged.

```python
# Feature selection
X = data[['Year', 'Month', 'Quarter']]
y = data['CSUSHPISA']
```

### 4. Train-Test Split

Split the dataset into training and testing sets to evaluate the model's performance effectively.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Ensemble Model Selection

We opt for an ensemble model, specifically the Random Forest Regressor. Random forests are known for their ability to capture complex relationships in data and provide accurate predictions.

```python # Ensemble Model Selection - Random Forest Regressor
model = RandomForestRegressor(random_state=42)
```
### 6. Hyperparameter Tuning

To optimize the model's performance, we employ hyperparameter tuning using GridSearchCV. This technique helps us find the best combination of hyperparameters for the Random Forest Regressor.

```python
# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_

# Get the best model
best_model = grid_search.best_estimator_
```

### 7. Model Training
Train the Random Forest Regressor on the training dataset using the selected hyperparameters.

```python
# Model Training
best_model.fit(X_train, y_train)
```

### 8. Model Evaluation
Evaluate the model's performance on the testing dataset using various metrics:

```python
# Model Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-squared (R2): {r2:.2f}')
```
### 9. Results and Insights

The model provides insights into how 'Year,' 'Month,' and 'Quarter' influence US home prices over the last 20 years. The metrics MSE, MAE, and R2 gauge the model's accuracy and explanatory power.

```
Mean Squared Error (MSE): 2.60
Mean Absolute Error (MAE): 0.88
R-squared (R2): 1.00
Best Hyperparameters: {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
```

### Visualize the model's predictions


### Conclusion

By implementing an ensemble model using a Random Forest Regressor with hyperparameter tuning, we can effectively analyze the impact of time-related factors on US home prices nationally. The model's performance metrics help us assess its accuracy and ability to explain variations in home prices.

This documentation provides a comprehensive overview of the entire process, from data acquisition to model evaluation, offering valuable insights for anyone interested in understanding and predicting US home prices.
