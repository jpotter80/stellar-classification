import pandas as pd

# Define paths
data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"

# Load the cleaned data
df = pd.read_csv(data_path)

# Check for null values
null_values = df.isnull().sum()

# Check data types
data_types = df.dtypes

# Print results
print("Null Values in Each Column:\n", null_values)
print("\nData Types of Each Column:\n", data_types)

# Save results to a file
validation_report_path = "/home/james/Documents/stellar-classification/validation_report.txt"
with open(validation_report_path, "w") as f:
    f.write("Null Values in Each Column:\n")
    f.write(null_values.to_string())
    f.write("\n\nData Types of Each Column:\n")
    f.write(data_types.to_string())

print(f"Validation report saved to {validation_report_path}")
