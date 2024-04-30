# clean.py
import pandas as pd
from load import load_dataset

def clean_data(df):
    """
    Function to clean data by converting columns to appropriate data types and handling missing values.
    Args:
    df (DataFrame): The pandas DataFrame to clean.

    Returns:
    DataFrame: The cleaned pandas DataFrame.
    """
    # Convert columns to numeric, setting errors to 'coerce' to handle non-numeric values
    df['Vmag'] = pd.to_numeric(df['Vmag'], errors='coerce')
    df['Plx'] = pd.to_numeric(df['Plx'], errors='coerce')
    df['e_Plx'] = pd.to_numeric(df['e_Plx'], errors='coerce')
    df['B-V'] = pd.to_numeric(df['B-V'], errors='coerce')

    # Drop rows where any of the above conversions resulted in NaN
    df.dropna(subset=['Vmag', 'Plx', 'e_Plx', 'B-V'], inplace=True)

    # Optional: Drop the 'Unnamed: 0' column if it's not needed
    df.drop(columns=['Unnamed: 0'], inplace=True)

    return df

if __name__ == "__main__":
    # Load the dataset
    df = load_dataset()
    # Clean the dataset
    df = clean_data(df)
    # Optionally, here you might save the cleaned data or pass it to another script for further processing
    print("Cleaned data:")
    print(df.head())  # Display the first few rows of the cleaned DataFrame
