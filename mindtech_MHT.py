# Project 1: Predicting Mental Health Treatment Needs (MHT)

# Reason for Model Choice: Random Forests handle both numerical and categorical data well, require minimal preprocessing,and provide interpretable feature importances. They are robust against overfitting due to their ensemble nature.

import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Setup logging
log_file = "model_training_mht.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting the script execution.")

# Load the dataset
# Load the CSV dataset using os to handle file paths dynamically. This ensures compatibility across systems,
# especially when the code is uploaded to GitHub or deployed in various environments.
data_path = os.path.join(os.getcwd(), 'survey.csv')
data = pd.read_csv(data_path)
print(data.head())
logging.info("Dataset loaded successfully.")

# Data Cleaning
# Fill missing values in 'self_employed' and 'work_interfere' columns with 'Unknown' to maintain data consistency.
logging.info("Filling missing values for 'self_employed' and 'work_interfere' columns.")
data['self_employed'] = data['self_employed'].fillna('Unknown')
data['work_interfere'] = data['work_interfere'].fillna('Unknown')

# Drop unnecessary columns
# Remove columns like 'Timestamp' and 'comments' as they are not relevant for the prediction task, reducing noise.
logging.info("Dropping unnecessary columns like 'Timestamp' and 'comments'.")
data = data.drop(columns=['Timestamp', 'comments'])

# Encode categorical variables
# Convert categorical data to numerical format using LabelEncoder, enabling compatibility with machine learning models.
logging.info("Encoding categorical variables using LabelEncoder.")
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
# Separate the dataset into features (X) and target variable (y) for supervised learning tasks.
logging.info("Splitting the dataset into features (X) and target (y).")
target = 'treatment'  # Target variable
X = data.drop(columns=[target])
y = data[target]

# Split into train and test sets
# Use an 80-20 split to divide the data into training and testing sets, ensuring a representative test evaluation.
logging.info("Splitting the data into training and testing sets.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
# Standardize numerical features using StandardScaler to improve model performance by normalizing feature values.
logging.info("Scaling numerical features using StandardScaler.")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection
# Initialize a Random Forest classifier, known for robustness and ability to handle both numerical and categorical data.
logging.info("Initializing the Random Forest classifier.")
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
# Perform hyperparameter tuning to identify the best combination of parameters for the Random Forest model.
logging.info("Starting hyperparameter tuning using GridSearchCV.")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
logging.info("GridSearchCV completed successfully.")

# Best model from grid search
# Extract the best estimator after grid search to use for final evaluation and testing.
best_rf = grid_search.best_estimator_
logging.info(f"Best parameters found: {grid_search.best_params_}")

# Evaluate on the test set
# Generate predictions on the test set and evaluate using classification metrics like accuracy and a detailed report.
logging.info("Evaluating the best model on the test set.")
y_pred = best_rf.predict(X_test)
classification_report_str = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Classification Report:\n{classification_report_str}")
logging.info(f"Accuracy: {accuracy}")

# Print results to the console
print("Classification Report:\n", classification_report_str)
print("Accuracy:", accuracy)

# Feature Importance
# Calculate and log feature importances to identify which features contribute most to model predictions.
logging.info("Calculating feature importance.")
feature_importances = best_rf.feature_importances_
important_features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
important_features = important_features.sort_values(by='Importance', ascending=False)
top_features_str = important_features.head(10).to_string()
logging.info(f"Top 10 Features:\n{top_features_str}")

# Print top features to the console
print("Top 10 Features:\n", top_features_str)

# Write logs and final outputs to a text file
# Save the best parameters, evaluation metrics, and important features to a text file for future reference.
output_file = os.path.join(os.getcwd(), "model_output_mht.txt")
with open(output_file, "w") as f:
    f.write("Random Forest Model Training and Evaluation\n")
    f.write("\nBest Parameters:\n")
    f.write(str(grid_search.best_params_))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report_str)
    f.write(f"\nAccuracy: {accuracy}\n")
    f.write("\nTop 10 Features:\n")
    f.write(top_features_str)

logging.info("Script execution completed. Results written to model_output_mht.txt.")

# Explanation of results
# The classification report shows precision, recall, F1-score, and support for each class, highlighting model performance.
# Accuracy reflects the percentage of correct predictions out of total predictions, offering an overall performance metric.
# Feature importance ranks the input variables based on their contribution to the model's predictive capability,
# helping to interpret the most influential factors in the prediction task.
