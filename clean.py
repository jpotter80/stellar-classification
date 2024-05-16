import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define paths
cleaned_data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"
viz_path = "/home/james/Documents/stellar-classification/viz/"

# Load the cleaned data
df = pd.read_csv(cleaned_data_path)

# Convert columns to numeric types
for column in ['Vmag', 'Plx', 'e_Plx', 'B-V']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Function to clean and impute missing values
def clean_and_impute(df):
    # Fill missing values with median
    df['Vmag'].fillna(df['Vmag'].median(), inplace=True)
    df['Plx'].fillna(df['Plx'].median(), inplace=True)
    df['e_Plx'].fillna(df['e_Plx'].median(), inplace=True)
    df['B-V'].fillna(df['B-V'].median(), inplace=True)
    
    # Handle outliers (example: cap Plx values at 99th percentile)
    plx_cap = df['Plx'].quantile(0.99)
    df['Plx'] = df['Plx'].clip(upper=plx_cap)
    
    return df

# Clean and impute the data
df = clean_and_impute(df)

# Function to create and save visualizations
def create_visualizations(df, viz_path):
    sns.set(style="whitegrid")

    # Plot the distribution of Vmag for Giants and Dwarfs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Vmag', hue='StarType', kde=True)
    plt.title('Distribution of Vmag by StarType')
    plt.savefig(os.path.join(viz_path, 'vmag_distribution.png'))
    plt.close()

    # Plot the distribution of Plx for Giants and Dwarfs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Plx', hue='StarType', kde=True)
    plt.title('Distribution of Plx by StarType')
    plt.savefig(os.path.join(viz_path, 'plx_distribution.png'))
    plt.close()

    # Pairplot of the features colored by StarType
    sns.pairplot(df[['Vmag', 'Plx', 'e_Plx', 'B-V', 'StarType']])
    plt.savefig(os.path.join(viz_path, 'pairplot.png'))
    plt.close()

# Main function
def main():
    create_visualizations(df, viz_path)
    print("Visualizations created and saved.")

if __name__ == "__main__":
    main()
