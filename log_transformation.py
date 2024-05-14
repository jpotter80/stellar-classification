import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    # Strip any whitespace and convert numeric columns
    for col in ['Vmag', 'Plx', 'e_Plx', 'B-V']:
        df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')

    # Impute missing values for B-V and SpType
    df['B-V'] = df['B-V'].fillna(df['B-V'].mean())
    df['SpType'] = df['SpType'].fillna('Unknown')

    # Drop rows where e_Plx is above a threshold (95th percentile)
    e_plx_threshold = df['e_Plx'].quantile(0.95)
    df = df[df['e_Plx'] <= e_plx_threshold]

    # Further handle outliers in Plx (99th percentile)
    plx_upper_limit = df['Plx'].quantile(0.99)
    df = df[df['Plx'] <= plx_upper_limit]

    # Ensure Plx and e_Plx are positive before applying log transformation
    df = df[df['Plx'] > 0]
    df = df[df['e_Plx'] > 0]

    # Apply log transformation to reduce skewness
    df['log_Plx'] = np.log1p(df['Plx'])
    df['log_e_Plx'] = np.log1p(df['e_Plx'])

    # Replace any remaining NaN or -inf values after log transformation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['log_Plx', 'log_e_Plx'], inplace=True)

    # Normalize using Min-Max scaling
    scaler = MinMaxScaler()
    df.loc[:, ['Vmag', 'B-V', 'log_Plx', 'log_e_Plx']] = scaler.fit_transform(df[['Vmag', 'B-V', 'log_Plx', 'log_e_Plx']])

    return df

def save_cleaned_data(df, filepath):
    """Save the cleaned DataFrame to a new CSV file."""
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    # Path to the raw CSV file
    data_path = '/home/james/Documents/stellar-classification/Star99999_raw.csv'
    # Path to save the cleaned CSV file
    cleaned_data_path = '/home/james/Documents/stellar-classification/Star99999_transformed.csv'

    # Load, clean, and save the data
    data = load_data(data_path)
    cleaned_data = clean_data(data)
    save_cleaned_data(cleaned_data, cleaned_data_path)