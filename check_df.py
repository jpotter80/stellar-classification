import pandas as pd

# Load the cleaned data
cleaned_data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"
df = pd.read_csv(cleaned_data_path)

# Display the DataFrame info
print(df.info())

# Display the first few rows of the DataFrame
print(df.head())
