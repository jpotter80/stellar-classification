import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Define paths
cleaned_data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"
model_path = "/home/james/Documents/stellar-classification/models/star_type_model.pkl"
metrics_path = "/home/james/Documents/stellar-classification/model_metrics.csv"

# Load the cleaned data
df = pd.read_csv(cleaned_data_path)

# Convert columns to numeric types
for column in ['Vmag', 'Plx', 'e_Plx', 'B-V']:
    df[column] = pd.to_numeric(df[column].str.strip(), errors='coerce')

# Drop rows with missing target values
df_filtered = df.dropna(subset=['StarType'])

# Update features and target
features = df_filtered[['Vmag', 'Plx', 'e_Plx', 'B-V']]
target = df_filtered['StarType']

# Fill missing values in features with median
features = features.fillna(features.median())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Perform hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
class_report = classification_report(y_test, predictions, output_dict=True)
conf_matrix = confusion_matrix(y_test, predictions)

# Save the best model
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# Save evaluation metrics
metrics = {
    "accuracy": [accuracy],
    "precision_dwarf": [class_report['Dwarf']['precision']],
    "recall_dwarf": [class_report['Dwarf']['recall']],
    "f1_dwarf": [class_report['Dwarf']['f1-score']],
    "precision_giant": [class_report['Giant']['precision']],
    "recall_giant": [class_report['Giant']['recall']],
    "f1_giant": [class_report['Giant']['f1-score']],
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(metrics_path, index=False)
print(f"Model metrics saved to {metrics_path}")

# Print evaluation results
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
