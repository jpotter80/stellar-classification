import pandas as pd
import os

# Define paths
data_path = "/home/james/Documents/stellar-classification/Star99999_raw.csv"
output_path = "/home/james/Documents/stellar-classification/unique_sptypes.csv"

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_unique_values(df, column):
    """Extract unique values from a specified column."""
    try:
        unique_values = df[column].unique()
        print(f"Extracted {len(unique_values)} unique values from column '{column}'")
        return unique_values
    except Exception as e:
        print(f"Error extracting unique values: {e}")
        return None

def save_unique_values(values, output_file):
    """Save unique values to a CSV file."""
    try:
        pd.DataFrame(values, columns=['SpType']).to_csv(output_file, index=False)
        print(f"Unique values saved to {output_file}")
    except Exception as e:
        print(f"Error saving unique values: {e}")

def main():
    # Load the dataset
    df = load_data(data_path)
    
    if df is not None:
        # Extract unique SpType values
        unique_sptypes = extract_unique_values(df, 'SpType')
        
        if unique_sptypes is not None:
            # Save unique values to CSV
            save_unique_values(unique_sptypes, output_path)

if __name__ == "__main__":
    main()
