import pandas as pd
import os

# Define paths
data_path = "/home/james/Documents/stellar-classification/Star99999_raw.csv"
output_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"

# Define the mapping based on the Wikipedia classification
def classify_star_type(sptype):
    if pd.isna(sptype):
        return None
    if any(lum in sptype for lum in ['I', 'II', 'III', 'IV']):
        return 'Giant'
    if 'V' in sptype or 'D' in sptype:
        return 'Dwarf'
    return None

def clean_data(df):
    """Clean the dataset and classify SpType."""
    try:
        # Apply the classification mapping
        df['StarType'] = df['SpType'].apply(classify_star_type)
        print(f"StarType column created with classifications.")
        return df
    except Exception as e:
        print(f"Error in cleaning data: {e}")
        return None

def save_cleaned_data(df, output_file):
    """Save the cleaned data to a CSV file."""
    try:
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

def main():
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Clean the dataset
    cleaned_df = clean_data(df)
    
    if cleaned_df is not None:
        # Save the cleaned data
        save_cleaned_data(cleaned_df, output_path)

if __name__ == "__main__":
    main()
