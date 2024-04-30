# save.py
from load import load_dataset  # Importing the function to load the dataset
from clean import clean_data   # Importing the function to clean the dataset

def save_data(df, file_path):
    """
    Saves the DataFrame to a CSV file.
    Args:
    df (DataFrame): The pandas DataFrame to save.
    file_path (str): The file path where the CSV will be saved.

    Returns:
    None: Saves the file to the specified location.
    """
    df.to_csv(file_path, index=False)  # Save DataFrame to CSV without the index

if __name__ == "__main__":
    df = load_dataset()  # Load the dataset using the function from load.py
    df = clean_data(df)  # Clean the dataset using the function from clean.py
    save_data(df, "C:/Users/Jamie/datasets/stellar-classification/Star99999_cleaned.csv")  # Specify path and filename for the cleaned data
