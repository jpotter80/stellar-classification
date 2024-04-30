import pandas as pd

# Load the dataset
df = pd.read_csv('C:/Users/Jamie/datasets/stellar-classification/Star99999_processed.csv')

# Print the names of all columns in the dataset
print("Features available in the dataset:")
print(df.columns.tolist())
