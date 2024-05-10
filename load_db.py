import os
import pandas as pd
from sqlalchemy import create_engine, Column, Float, Integer, String, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError

# Using environment variables to configure the database connection URI
DATABASE_USER = os.getenv('DB_USER')
DATABASE_PASSWORD = os.getenv('DB_PASS')
DATABASE_HOST = os.getenv('DB_HOST')
DATABASE_NAME = 'star_db'

# Connection string for the star_db database
DATABASE_URI = f'postgresql+pg8000://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}'

def create_table_and_load_data():
    """Create the table and load data into the PostgreSQL database if it does not exist."""
    engine = create_engine(DATABASE_URI)
    metadata = MetaData()
    
    # Define the table schema
    stellar_data = Table('stellar_data', metadata,
                         Column('Vmag', Float),
                         Column('Plx', Float),
                         Column('e_Plx', Float),
                         Column('B-V', Float),
                         Column('SpType', String),
                         Column('cluster', String),
                         Column('Spectral_Class', String),
                         Column('Spectral_Class_Code', Integer),
                         Column('Spectral_Class_(', Integer),
                         Column('Spectral_Class_A', Integer),
                         Column('Spectral_Class_B', Integer),
                         Column('Spectral_Class_C', Integer),
                         Column('Spectral_Class_D', Integer),
                         Column('Spectral_Class_F', Integer),
                         Column('Spectral_Class_G', Integer),
                         Column('Spectral_Class_K', Integer),
                         Column('Spectral_Class_M', Integer),
                         Column('Spectral_Class_N', Integer),
                         Column('Spectral_Class_O', Integer),
                         Column('Spectral_Class_R', Integer),
                         Column('Spectral_Class_S', Integer),
                         Column('Spectral_Class_W', Integer),
                         Column('Spectral_Class_k', Integer),
                         Column('Spectral_Class_s', Integer),
                         Column('Spectral_Class_nan', Integer),
                         Column('PCA1', Float),
                         Column('PCA2', Float))
    
    try:
        metadata.create_all(engine)
        print("Table 'stellar_data' created successfully!")
        
        # Load data from CSV
        data = pd.read_csv('/home/james/Documents/stellar-classification/Star99999_encoded.csv')
        data.to_sql('stellar_data', con=engine, if_exists='append', index=False)
        print("Data loaded successfully into 'stellar_data'.")
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    create_table_and_load_data()
