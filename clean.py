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

    # Drop rows where e_Plx is above a threshold
    e_plx_threshold = df['e_Plx'].quantile(0.95)  # using 95th percentile as threshold
    df = df[df['e_Plx'] <= e_plx_threshold]

    # Normalize using Min-Max scaling
    scaler = MinMaxScaler()
    df.loc[:, ['Vmag', 'Plx', 'B-V']] = scaler.fit_transform(df[['Vmag', 'Plx', 'B-V']])  # Use .loc to avoid SettingWithCopyWarning

    return df

def save_cleaned_data(df, filepath):
    """Save the cleaned DataFrame to a new CSV file."""
    df.to_csv(filepath, index=False)

# Correct path to your CSV file
data_path = '/home/james/Documents/stellar-classification/Star99999_raw.csv'
cleaned_data_path = '/home/james/Documents/stellar-classification/Star99999_cleaned.csv'

# Using the functions
data = load_data(data_path)
cleaned_data = clean_data(data)
save_cleaned_data(cleaned_data, cleaned_data_path)
