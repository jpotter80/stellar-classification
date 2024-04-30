import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Load the data
df = pd.read_csv('C:/Users/Jamie/datasets/stellar-classification/Star99999_processed.csv')

# Define spectral classes in order of temperature
spectral_order = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

# Extract the first letter of each spectral type as its class
df['Spectral_Class'] = df['SpType'].str[0]

# Ordinal encoding for spectral classes
df['Spectral_Class_Code'] = df['Spectral_Class'].apply(lambda x: spectral_order.index(x) if x in spectral_order else None)

# Optional: One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
spectral_one_hot = encoder.fit_transform(df[['Spectral_Class']])
spectral_one_hot_df = pd.DataFrame(spectral_one_hot, columns=encoder.get_feature_names_out(['Spectral_Class']))
df = pd.concat([df, spectral_one_hot_df], axis=1)

# Optional: PCA for dimensionality reduction on one-hot encoded features
pca = PCA(n_components=2)
spectral_pca = pca.fit_transform(spectral_one_hot)
df_pca = pd.DataFrame(spectral_pca, columns=['PCA1', 'PCA2'])
df = pd.concat([df, df_pca], axis=1)

# Normalize numeric data
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Define the file path
file_path = 'C:/Users/Jamie/stellar-classification/Star99999_encoded.csv'

# Save the preprocessed data
try:
    df.to_csv(file_path, index=False)
    print(f"Data encoding and preprocessing complete, file saved at {file_path}")
except Exception as e:
    print(f"Failed to save the file: {str(e)}")

# Check if the file exists
if os.path.exists(file_path):
    print(f"File successfully saved at {file_path}")
else:
    print(f"File not found at {file_path}. Please check directory permissions and disk space.")
