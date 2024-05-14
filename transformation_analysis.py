import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_transformed_data(filepath, save_dir):
    """Analyze the transformed dataset."""
    df = pd.read_csv(filepath)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Basic information
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

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(save_dir, 'correlation_matrix_transformed.png'))
    plt.close()
    
    # Distribution plots for each numeric feature
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(save_dir, f'distribution_{col}_transformed.png'))
        plt.close()

# Path to the transformed CSV file
data_path = 'Star99999_transformed.csv'
# Path to save the visualizations
save_dir = '/home/james/Documents/stellar-classification/viz/'

# Analyze the transformed data
analyze_transformed_data(data_path, save_dir)

