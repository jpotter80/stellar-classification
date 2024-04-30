# explore.py
from load import load_dataset  # Import the dataset loading function from load.py

def explore_data(df):
    """
    Function to perform basic exploratory data analysis on a DataFrame.
    Args:
    df (DataFrame): The pandas DataFrame to explore.

    Prints:
    Various outputs from exploratory data analysis.
    """
    print("First few rows of the dataset:")
    print(df.head())  # Print the first five rows of the DataFrame
    
    print("\nData types of each column:")
    print(df.dtypes)  # Print the data types of each column
    
    print("\nSummary of statistics for numeric columns:")
    print(df.describe())  # Print descriptive statistics for numerical columns
    
    print("\nNumber of missing values in each column:")
    print(df.isnull().sum())  # Print the sum of missing values in each column
    
    print("\nShape of the DataFrame (rows, columns):")
    print(df.shape)  # Print the number of rows and columns in the DataFrame

if __name__ == "__main__":
    df = load_dataset()  # Load the dataset using the function from load.py
    explore_data(df)  # Call the explore_data function with the loaded DataFrame
