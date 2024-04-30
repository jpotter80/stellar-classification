import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Define paths for outputs
image_dir = 'C:/Users/Jamie/stellar-classification/viz/'
file_dir = 'C:/Users/Jamie/stellar-classification/'

# Load the predictions and the original dataset
df_predictions = pd.read_csv(file_dir + 'model_predictions.csv')
df = pd.read_csv(file_dir + 'Star99999_processed.csv')

# Handle inf values explicitly
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_predictions.replace([np.inf, -np.inf], np.nan, inplace=True)

# Display basic evaluation metrics
print(classification_report(df_predictions['Actual'], df_predictions['Predicted']))
report = classification_report(df_predictions['Actual'], df_predictions['Predicted'], output_dict=False)

# Save the classification report to a text file
with open(file_dir + 'classification_report.txt', 'w') as f:
    f.write(report)

# Generate and plot the confusion matrix
conf_mat = confusion_matrix(df_predictions['Actual'], df_predictions['Predicted'])
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(image_dir + 'confusion_matrix.png')
plt.close()  # Close the plot to avoid display

# Merge the prediction results with the original dataset
df_merged = pd.merge(df_predictions, df, left_index=True, right_index=True, how='left')

# Exploring each feature
features = ['Vmag', 'Plx', 'e_Plx', 'B-V', 'SpType', 'cluster']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature].dropna(), kde=True, bins=30)  # Ensure no NaN values for plotting
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(image_dir + f'{feature}_distribution.png')
    plt.close()  # Close the plot to avoid display

    sns.boxplot(x='Actual', y=feature, hue='Predicted', data=df_merged)
    plt.title(f'Error Analysis by {feature}')
    plt.savefig(image_dir + f'{feature}_error_analysis.png')
    plt.close()  # Close the plot to avoid display

    # Save detailed feature error analysis to a CSV
    df_feature_errors = df_merged[df_merged['Actual'] != df_merged['Predicted']]
    df_feature_errors[[feature, 'Actual', 'Predicted']].to_csv(file_dir + f'{feature}_errors.csv', index=False)

print("Analysis complete and files saved.")
