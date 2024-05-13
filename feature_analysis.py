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

    # Handle single feature for SHAP summary plot
    shap_values = shap_values[1]  # for binary classification shap_values[1] holds the values for the positive class
    feature_names = df_features.columns.tolist()
    
    # Check if there's only one feature and adjust plotting
    if len(feature_names) == 1:
        # Plotting SHAP values for a single feature
        plt.figure()
        plt.title("SHAP Values for Predicted Feature")
        plt.bar(feature_names, shap_values.mean(axis=0))
        plt.xlabel("Feature")
        plt.ylabel("SHAP Value (mean impact)")
        plt.savefig(output_image_path)
        print(f"SHAP summary plot for single feature saved to {output_image_path}.")
    else:
        # Normal SHAP summary plot
        shap.summary_plot(shap_values, df_features, feature_names=feature_names, show=False)
        plt.savefig(output_image_path)
        print(f"SHAP analysis complete and summary plot saved to {output_image_path}.")

    # Optional: Save SHAP values to a CSV file
    np.savetxt(output_csv_path, shap_values, delimiter=',')
    print(f"SHAP values saved to {output_csv_path}.")
else:
    print("Error: 'Actual' column missing or no additional features available for analysis.")
