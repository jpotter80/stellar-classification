import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text

# Using environment variables to configure the database connection URI
DATABASE_USER = os.getenv('DB_USER')
DATABASE_PASSWORD = os.getenv('DB_PASS')
DATABASE_HOST = os.getenv('DB_HOST')

# Connection string for the default PostgreSQL database
DEFAULT_DATABASE_URI = f'postgresql+pg8000://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}/postgres'

def create_database(db_name):
    """Create the PostgreSQL database if it does not exist."""
    engine = create_engine(DEFAULT_DATABASE_URI, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        try:
            # Check if the database does not exist and then create it
            exists = conn.scalar(text(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = :db"), {'db': db_name})
            if not exists:
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"Database '{db_name}' created successfully!")
            else:
                print(f"Database '{db_name}' already exists.")
        except SQLAlchemyError as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    create_database('star_db')
