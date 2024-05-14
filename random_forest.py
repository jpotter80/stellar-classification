import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def save_model(model, filepath):
    """Save the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def plot_feature_importance(model, features, filepath):
    """Plot feature importance and save the plot."""
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance for Random Forest Model')
    plt.savefig(filepath)
    print(f"Feature importance plot saved to {filepath}")

if __name__ == "__main__":
    # Path to the processed CSV file
    data_path = '/home/james/Documents/stellar-classification/Star99999_transformed.csv'
    # Path to save the trained model
    model_path = '/home/james/Documents/stellar-classification/models/random_forest_model.pkl'
    # Path to save the feature importance plot
    feature_importance_path = '/home/james/Documents/stellar-classification/viz/feature_importance.png'
    # Paths to save model performance and predictions
    metrics_path = '/home/james/Documents/stellar-classification/model_performance.txt'
    predictions_path = '/home/james/Documents/stellar-classification/model_predictions.csv'

    # Load the data
    df = load_data(data_path)

    # Define the target variable and features
    df['is_giant'] = df['SpType'].apply(lambda x: 1 if 'III' in str(x) else 0 if 'V' in str(x) else None)
    df = df.dropna(subset=['is_giant'])
    df['is_giant'] = df['is_giant'].astype(int)
    X = df.drop(['is_giant', 'SpType'], axis=1)
    y = df['is_giant']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Output the performance metrics
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Save the model to disk
    save_model(clf, model_path)

    # Save the performance metrics to a text file
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
        f.write("Classification Report:\n")
        f.write(f"{class_report}\n")
    print(f"Performance metrics saved to {metrics_path}")

    # Save the predictions to a CSV file
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Plot and save feature importance
    plot_feature_importance(clf, X.columns, feature_importance_path)
