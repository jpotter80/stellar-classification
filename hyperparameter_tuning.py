import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def save_model(model, filepath):
    """Save the model to a file."""
    joblib.dump(model, filepath)

def plot_feature_importance(model, features, filepath):
    """Plot and save the feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Random Forest Model')
    plt.savefig(filepath)
    plt.show()

if __name__ == "__main__":
    # Path to the cleaned and transformed CSV file
    data_path = '/home/james/Documents/stellar-classification/Star99999_transformed.csv'

    # Load the data
    df = load_data(data_path)

    # Define the target variable
    df['is_giant'] = df['SpType'].apply(lambda x: 1 if 'III' in x else (0 if 'V' in x else None))
    df = df.dropna(subset=['is_giant'])
    df['is_giant'] = df['is_giant'].astype(int)

    # Features and target
    X = df.drop(['is_giant', 'SpType'], axis=1)
    y = df['is_giant']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Set up the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_

    # Predict and evaluate the model
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Output the performance metrics
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Save the model to disk
    model_path = '/home/james/Documents/stellar-classification/models/best_random_forest_model.pkl'
    save_model(best_rf, model_path)
    print(f"Model saved to {model_path}")

    # Save the performance metrics to a text file
    metrics_path = '/home/james/Documents/stellar-classification/model_performance_best.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
        f.write("Classification Report:\n")
        f.write(f"{class_report}\n")
    print(f"Performance metrics saved to {metrics_path}")

    # Save the predictions to a CSV file
    predictions_path = '/home/james/Documents/stellar-classification/model_predictions_best.csv'
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Plot and save feature importance
    feature_importance_path = '/home/james/Documents/stellar-classification/viz/feature_importance_best.png'
    plot_feature_importance(best_rf, X.columns, feature_importance_path)
    print(f"Feature importance plot saved to {feature_importance_path}")
