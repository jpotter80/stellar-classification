import pandas as pd
from load import load_dataset  # Import the dataset loading function from load.py
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    """
    Perform basic exploratory data analysis on a DataFrame.
    """
    print("First few rows of the dataset:")
    print(df.head())
    
    print("\nData types of each column:")
    print(df.dtypes)
    
    print("\nSummary of statistics for numeric columns:")
    print(df.describe())
    
    print("\nNumber of missing values in each column:")
    print(df.isnull().sum())
    
    print("\nShape of the DataFrame (rows, columns):")
    print(df.shape)

    # Exclude non-numeric columns for skewness and kurtosis calculations
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    print("\nSkewness of numeric data:")
    print(df[numeric_cols].skew())

    print("\nKurtosis of numeric data:")
    print(df[numeric_cols].kurtosis())

def normalize_and_plot(df):
    """
    Apply normalization to the dataframe and plot before and after results for comparison.
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
    plt.savefig('/home/james/Documents/stellar-classification/viz/normalized_comparison.png')  # Save plot to the specified directory
    plt.close()

def detailed_eda(df):
    """
    Perform detailed exploratory data analysis and create visualizations.
    """
    # Correlation matrix
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.savefig('/home/james/Documents/stellar-classification/viz/correlation_matrix.png')
    plt.close()
    
    # Pair plot
    sns.pairplot(df[numeric_cols])
    plt.savefig('/home/james/Documents/stellar-classification/viz/pair_plot.png')
    plt.close()

    # Distribution plots for each numeric feature
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'/home/james/Documents/stellar-classification/viz/distribution_{col}.png')
        plt.close()

if __name__ == "__main__":
    # Path to the cleaned CSV file
    data_path = '/home/james/Documents/stellar-classification/Star99999_cleaned.csv'
    
    df = load_dataset(data_path)  # Load the dataset using the function from load.py
    explore_data(df)  # Perform basic EDA
    normalize_and_plot(df)  # Normalize and plot data
    detailed_eda(df)  # Perform detailed EDA and create visualizations
