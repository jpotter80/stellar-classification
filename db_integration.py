import os
import pandas as pd
from sqlalchemy import create_engine

# Retrieve database credentials from environment variables
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')

# Database connection using pg8000 with environment variables for credentials
db_uri = f"postgresql+pg8000://{db_user}:{db_pass}@{db_host}:{db_port}/star_db"
engine = create_engine(db_uri)

# Load the cleaned data
data_path = "/home/james/Documents/stellar-classification/cleaned_data.csv"
df = pd.read_csv(data_path)

# Load predictions
predictions_path = "/home/james/Documents/stellar-classification/predictions.csv"
predictions = pd.read_csv(predictions_path)

# Save cleaned data to database
df.to_sql('cleaned_star_data', engine, if_exists='replace', index=False)

# Save predictions to database
predictions.to_sql('predicted_star_data', engine, if_exists='replace', index=False)

print("Data and predictions have been saved to the database.")
