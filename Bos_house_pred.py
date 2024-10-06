import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv('HousingData.csv')

# Handle missing values: Fill missing values with the median of each column
data.fillna(data.median(), inplace=True)

# Separate features and target
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3,5,7,10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Calculate and print the R² score
r2 = r2_score(y_test, y_pred)
print(f"R² score: {r2}")

# Save the model to a file
joblib.dump(best_rf_model, 'random_forest_model.pkl')
print("Model saved to random_forest_model.pkl")

# Feature importance
importances = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save feature importances to CSV
feature_importance_df.to_csv('feature_importances.csv', index=False)
print("Feature importances saved to feature_importances.csv")

# Display feature importances
print("Feature importances:")
print(feature_importance_df)

# Plot Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_rf_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='r2', n_jobs=-1)

# Calculate mean and standard deviation for training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.title('Learning Curve for Random Forest Regressor')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend(loc="best")
plt.grid()
plt.show()

# Take user input for prediction at the end
user_input = {}
for column in X.columns:
    user_input[column] = float(input(f"Enter value for {column}: "))

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Predict the price based on user input
user_pred = best_rf_model.predict(user_input_df)
print(f"Predicted price based on user input: {user_pred[0]}")
