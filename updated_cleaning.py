import pandas as pd

# Define paths
data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"
validation_report_path = "/home/james/Documents/stellar-classification/validation_report.txt"

# Load the cleaned data
df = pd.read_csv(data_path)

# Convert columns to numeric types
for column in ['Vmag', 'Plx', 'e_Plx', 'B-V']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Impute missing values in numeric columns
df['Vmag'].fillna(df['Vmag'].median(), inplace=True)
df['Plx'].fillna(df['Plx'].median(), inplace=True)
df['e_Plx'].fillna(df['e_Plx'].median(), inplace=True)
df['B-V'].fillna(df['B-V'].median(), inplace=True)

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

# Save cleaned data
cleaned_data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"
df.to_csv(cleaned_data_path, index=False)
print(f"Cleaned data saved to {cleaned_data_path}")
