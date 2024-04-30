# load_data.py
import pandas as pd  # Importing the pandas library for data manipulation

def load_dataset():
    """
    Function to load a dataset from a specific CSV file.
    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    file_path = "C:/Users/Jamie/datasets/stellar-classification/Star99999_raw.csv"  # Specify the path to your dataset here
    df = pd.read_csv(file_path)  # Load the CSV file into a pandas DataFrame
    return df  # Return the DataFrame to the caller
