import pandas as pd
import os

# Define file path
data_path = "/home/james/Documents/stellar-classification/Star99999_transformed.csv"

# Check if the data file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file {data_path} does not exist.")
else:
    print(f"Data file found at {data_path}")

# Load your dataset
df = pd.read_csv(data_path)
print("Available columns in the dataset:")
print(df.columns)
