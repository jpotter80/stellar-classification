import os
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.schema import DropConstraint, DropTable, CreateTable
from sqlalchemy import text

# Retrieve database credentials from environment variables
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')

# Database connection using pg8000 with environment variables for credentials
db_uri = f"postgresql+pg8000://{db_user}:{db_pass}@{db_host}:{db_port}/star_db"
engine = create_engine(db_uri)

# Function to rename the first column of a table
def rename_first_column(engine, table_name, old_column_name, new_column_name):
    with engine.connect() as conn:
        # Begin a transaction
        conn.execute(text("BEGIN"))

        # Rename the column
        conn.execute(text(f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_column_name}" TO "{new_column_name}";'))

        # Commit the transaction
        conn.execute(text("COMMIT"))

# Rename the first column in 'cleaned_star_data' table
rename_first_column(engine, 'cleaned_star_data', 'Unnamed: 0', 'Id')

# Rename the first column in 'predicted_star_data' table
rename_first_column(engine, 'predicted_star_data', 'Unnamed: 0', 'Id')

print("Column names updated successfully.")
