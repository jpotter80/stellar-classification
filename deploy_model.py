import joblib
import pandas as pd

# Define paths
model_path = "/home/james/Documents/stellar-classification/models/star_type_model.pkl"
data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"
predictions_path = "/home/james/Documents/stellar-classification/predictions.csv"
validation_report_path = "/home/james/Documents/stellar-classification/validation_report.txt"

# Load the cleaned data
df = pd.read_csv(data_path)

# Check for null values and data types
null_values = df.isnull().sum()
data_types = df.dtypes

# Save validation report
with open(validation_report_path, "w") as f:
    f.write("Null Values in Each Column:\n")
    f.write(null_values.to_string())
    f.write("\n\nData Types of Each Column:\n")
    f.write(data_types.to_string())

print(f"Validation report saved to {validation_report_path}")

# Ensure only numeric columns are filled with the median
numeric_columns = ['Vmag', 'Plx', 'e_Plx', 'B-V']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Load the model
model = joblib.load(model_path)

# Make predictions
df['Predicted_StarType'] = model.predict(df[numeric_columns])

# Save predictions
df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")
