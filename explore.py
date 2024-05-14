# explore.py
import pandas as pd
from load import load_dataset  # Import the dataset loading function from load.py
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    """
    Function to perform basic exploratory data analysis on a DataFrame.
    Args:
    df (DataFrame): The pandas DataFrame to explore.
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

    # Exclude non-numeric columns for skewness and kurtosis calculations
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    print("\nSkewness of numeric data:")
    print(df[numeric_cols].skew())

    print("\nKurtosis of numeric data:")
    print(df[numeric_cols].kurtosis())

def normalize_and_plot(df):
    """
    Apply normalization to the dataframe and plot before and after results for comparison.
    Args:
    df (DataFrame): The pandas DataFrame to normalize and plot.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
    
    # Plotting original vs normalized data
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.boxplot(data=df[numeric_cols], ax=axes[0])
    axes[0].set_title('Original Data')
    sns.boxplot(data=df_normalized, ax=axes[1])
    axes[1].set_title('Normalized Data')
    plt.show()

if __name__ == "__main__":
    # Correct path to your CSV file
    data_path = '/home/james/Documents/stellar-classification/Star99999_cleaned.csv'
    
    df = load_dataset(data_path)  # Load the dataset using the function from load.py with the correct filename
    explore_data(df)  # Call the explore_data function with the loaded DataFrame
    normalize_and_plot(df)  # Normalize and plot data
