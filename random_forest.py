import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the processed dataset
df = pd.read_csv('C:/Users/Jamie/datasets/stellar-classification/Star99999_processed.csv')

# Define a function to classify giants and dwarfs
def classify_giant_dwarf(sptype):
    # Check if sptype is a string and not NaN
    if isinstance(sptype, str):
        if 'III' in sptype:
            return 1  # Represents a giant
        elif 'V' in sptype:
            return 0  # Represents a dwarf
    return None  # For non-string or NaN values

# Apply the function to create a new target column
df['is_giant'] = df['SpType'].apply(classify_giant_dwarf)

# Drop rows where 'is_giant' is None if there are any
df = df.dropna(subset=['is_giant'])

# Convert 'is_giant' to an integer type
df['is_giant'] = df['is_giant'].astype(int)

X = df.drop(['is_giant', 'SpType', 'cluster'], axis=1)  # Drop non-numeric and target columns
y = df['is_giant']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the performance metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save the model to disk
model_path = 'C:/Users/Jamie/datasets/stellar-classification/random_forest_model.joblib'
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Save the performance metrics to a text file
metrics_path = 'C:/Users/Jamie/datasets/stellar-classification/model_performance.txt'
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{conf_matrix}\n")
    f.write("Classification Report:\n")
    f.write(f"{class_report}\n")
print(f"Performance metrics saved to {metrics_path}")

# Save the predictions to a CSV file
predictions_path = 'C:/Users/Jamie/datasets/stellar-classification/model_predictions.csv'
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")
