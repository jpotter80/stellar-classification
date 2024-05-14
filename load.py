import pandas as pd
import os

def load_dataset(filename):
    """
    Function to load a dataset from a specified CSV file.
    Args:
    filename (str): The name of the CSV file to be loaded.
    
    Returns:
    DataFrame: A pandas DataFrame containing the loaded data.
    """
    # Build the full file path using the specified filename, ensuring it starts from the root directory
    file_path = os.path.join('/home', 'james', 'Documents', 'stellar-classification', filename)
    df = pd.read_csv(file_path)  # Load the CSV file into a pandas DataFrame
    return df  # Return the DataFrame to the caller

# Example usage
if __name__ == "__main__":
    # Specify the filename of the CSV file to be loaded
    data = load_dataset('Star99999_raw.csv')
    print(data.head())  # Display the first few rows of the DataFrame to verify loading
