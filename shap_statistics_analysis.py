
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

# Expand the tilde to the user's home directory
home_directory = os.path.expanduser('~')

# File paths for the project using the expanded home directory
shap_values_csv_path = os.path.join(home_directory, 'Documents/stellar-classification/shap_values.csv')

# Load the SHAP values
shap_values_df = pd.read_csv(shap_values_csv_path, header=None)
shap_values = shap_values_df[0].values  # Assuming SHAP values are in the first column

# Compute basic statistics
mean_value = np.mean(shap_values)
median_value = np.median(shap_values)
mode_result = mode(shap_values)
mode_value = 'No clear mode'  # Default if mode conditions are not met
if mode_result.count.size > 0 and mode_result.count[0] > 1:
    mode_value = mode_result.mode[0]

variance_value = np.var(shap_values)
std_dev_value = np.std(shap_values)

# Print the statistics
print("Descriptive Statistics of SHAP Values:")
print(f"Mean: {mean_value:.4f}")
print(f"Median: {median_value:.4f}")
print(f"Mode: {mode_value}")
print(f"Variance: {variance_value:.4f}")
print(f"Standard Deviation: {std_dev_value:.4f}")

# Plot the distribution of SHAP values
plt.figure(figsize=(10, 6))
plt.hist(shap_values, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of SHAP Values')
plt.xlabel('SHAP Value')
plt.ylabel('Frequency')
plt.grid(True)
output_image_path = os.path.join(home_directory, 'Documents/stellar-classification/viz/shap_values_distribution.png')
plt.savefig(output_image_path)
print(f"Distribution plot saved to {output_image_path}")
