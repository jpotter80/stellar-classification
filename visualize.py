# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from load import load_dataset

def visualize_data(df):
    output_dir = "C:/Users/Jamie/datasets/stellar-classification/viz/"
    
    # Handle inf and -inf explicitly, replacing them with NaN
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)  # Optionally drop all rows with NaN to ensure clean data for visualization

    # Histogram for Vmag
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Vmag'], bins=30, kde=True)
    plt.title('Distribution of Vmag')
    plt.xlabel('Vmag')
    plt.ylabel('Frequency')
    plt.savefig(output_dir + 'Vmag_Distribution.png')
    plt.close()

    # Boxplot for B-V Color Index across Spectral Types
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='SpType', y='B-V', data=df)
    plt.title('B-V Color Index by Spectral Type')
    plt.xlabel('Spectral Type')
    plt.ylabel('B-V Color Index')
    plt.savefig(output_dir + 'BV_Color_Index_by_Spectral_Type.png')
    plt.close()

    # Scatter plot for B-V Color Index vs Vmag
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='B-V', y='Vmag', data=df, hue='SpType', alpha=0.6)
    plt.title('B-V Color Index vs. Vmag')
    plt.xlabel('B-V Color Index')
    plt.ylabel('Vmag')
    plt.savefig(output_dir + 'BV_vs_Vmag_Scatterplot.png')
    plt.close
