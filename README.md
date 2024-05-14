# Stellar Classification Project

## Overview
This project is dedicated to the classification of stars based on their spectral characteristics using the Morgan-Keenan (MK) system. The project utilizes Python for data analysis, covering data loading, cleaning, exploration, and normalization to prepare the dataset for machine learning applications.

## Project Structure

- `load.py`: Contains the function to load the dataset.
- `clean.py`: Script to clean the dataset by handling missing values and removing outliers.
- `explore.py`: Used for exploratory data analysis to understand dataset characteristics and normalize data.
- `README.md`: Provides documentation for the project setup and execution instructions.

## Data Description

The dataset used in this project classifies stars into various spectral types and luminosity classes. It includes several key features:
- `Vmag`: Visual Apparent Magnitude of the Star.
- `Plx`: Distance Between the Star and the Earth.
- `e_Plx`: Standard Error of Plx.
- `B-V`: B-V color index.
- `SpType`: Spectral type.
- `Amag`: Absolute Magnitude of the Star.

## File Paths

- Dataset path: `/home/james/Documents/stellar-classification/Star99999_raw.csv`
- Cleaned data path: `/home/james/Documents/stellar-classification/Star99999_cleaned.csv`

## Usage

### Setup Environment

Ensure that your Python environment is set up with the necessary libraries:
```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Running Scripts

1. **Loading Data**
   ```bash
   python load.py
   ```
   This script will load data from `/home/james/Documents/stellar-classification/Star99999_raw.csv`.

2. **Cleaning Data**
   ```bash
   python clean.py
   ```
   Executes the cleaning process and saves the cleaned data to `/home/james/Documents/stellar-classification/Star99999_cleaned.csv`.

3. **Exploring Data**
   ```bash
   python explore.py
   ```
   Performs exploratory data analysis on the cleaned data, providing insights into the dataset and visualizing the normalized data.

## Data Handling Strategies

- **Missing Data**: Missing values in `B-V` and `SpType` are addressed through imputation or removal based on the extent and nature of missingness.
- **Normalization**: Data normalization is performed to ensure that numerical features contribute equally to analytic models.

## Validation and Analysis

After cleaning and normalization, validate the methods by comparing the properties of the imputed and original data. Sensitivity analysis is recommended to assess the impact of the data handling strategies on the outcomes of the classification models.

## Contribution

Contributions to the project are welcome. Please ensure to update the documentation and paths as necessary to keep the project coherent and up-to-date.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
```

### Notes:
- **File Paths**: All file paths are updated to reflect the Linux system paths as you've provided.
- **Scripts and Usage**: Clear instructions on how to run each script and what each script does.
- **Data Handling Strategies**: Explanation of how missing data is treated and the role of normalization.
- **Validation and Analysis**: Encourages validating the changes made to the data handling process.

This README should provide a comprehensive guide to maintaining and using the project effectively.