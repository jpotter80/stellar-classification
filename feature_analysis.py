import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Expand the tilde to the user's home directory
home_directory = os.path.expanduser('~')

# File paths for the project using the expanded home directory
input_csv_path = os.path.join(home_directory, 'Documents/stellar-classification/model_predictions.csv')
output_image_path = os.path.join(home_directory, 'Documents/stellar-classification/viz/shap_summary_plot.png')
output_csv_path = os.path.join(home_directory, 'Documents/stellar-classification/shap_values.csv')

# Load the scaled data
df_scaled = pd.read_csv(input_csv_path)
print("Data loaded successfully.")
print("Data shape (rows, columns):", df_scaled.shape)

# Check the first few rows to ensure data is loaded correctly
print("First few rows of data:", df_scaled.head())

# Ensure the 'Actual' column exists and is not the only column
if 'Actual' in df_scaled.columns and len(df_scaled.columns) > 1:
    y = df_scaled['Actual']
    df_features = df_scaled.drop(columns=['Actual'])

    # Fit a model to use with SHAP
    model = RandomForestClassifier(random_state=42)
    model.fit(df_features, y)

    # SHAP values using the fitted model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_features)

    # Flatten SHAP values for the positive class and plot
    shap_values_flat = shap_values[1][:, 0]  # Select the SHAP values for the positive class and flatten

    print("Length of SHAP values:", len(shap_values_flat))
    print("Example SHAP values:", shap_values_flat[:5])

    # Plotting individual SHAP values for each prediction
    plt.figure(figsize=(10, 5))
    plt.title("SHAP Values for Each Prediction", fontsize=16)
    plt.scatter(range(len(shap_values_flat)), shap_values_flat, color='blue', alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.axhline(y=np.mean(shap_values_flat), color='red', linestyle='-', linewidth=2, label=f'Mean SHAP Value: {np.mean(shap_values_flat):.4f}')
    plt.xlabel("Prediction Index", fontsize=14)
    plt.ylabel("SHAP Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(output_image_path)
    print(f"Updated SHAP summary plot saved to {output_image_path}.")

    # Optional: Save SHAP values to a CSV file
    np.savetxt(output_csv_path, shap_values_flat, delimiter=',')
    print(f"SHAP values saved to {output_csv_path}.")
else:
    print("Error: 'Actual' column missing or no additional features available for analysis.")
