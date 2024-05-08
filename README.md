
# Stellar Classification Project

## Project Overview
This project aims to classify stars into giants and dwarfs using spectral data. We employ machine learning models, specifically a Random Forest classifier, to predict the classification based on features derived from spectral measurements.

## Features
The dataset includes the following features used for classification:
- `Vmag`: Visual magnitude of the star.
- `Plx`: Parallax of the star, giving an indication of the distance.
- `e_Plx`: Error in parallax measurement.
- `B-V`: The color index of the star.
- `SpType`: The spectral type of the star.
- `cluster`: Assigned cluster from preliminary clustering analysis.

## Getting Started

### Prerequisites
- Python 3.x
- PostgreSQL
- Poetry for dependency management

### Installation
Clone the repository and navigate to the project directory. Set up the environment and install the required dependencies using Poetry:

```bash
poetry install
```

### Database Setup
Ensure PostgreSQL is running and the `star_db` database is accessible. Use the `database.py` script to set up any necessary database schemas:

```bash
poetry run python database.py
```

### File Descriptions
- `database.py`: Configures the PostgreSQL database for storing and retrieving data.
- `load.py`: Loads data from the CSV file into the database.
- `clean.py`: Cleans and preprocesses the data within the database.
- `explore.py`: Performs exploratory data analysis on the dataset.
- `visualize.py`: Generates visual plots to explore the data visually.
- `save.py`: Saves the cleaned data in the database for further use.
- `clustering_analysis.py`: Performs clustering to find patterns in the data.
- `random_forest.py`: Implements the Random Forest classifier for star classification.

## Usage

### Running the Scripts
To run the scripts within the Poetry environment, navigate to the project directory and execute the following commands:

1. **Setting up the database**
   ```bash
   poetry run python database.py
   ```

2. **Loading and cleaning data**
   ```bash
   poetry run python load.py
   poetry run python clean.py
   ```

3. **Exploratory Data Analysis**
   ```bash
   poetry run python explore.py
   ```

4. **Data Visualization**
   ```bash
   poetry run python visualize.py
   ```

5. **Clustering Analysis**
   ```bash
   poetry run python clustering_analysis.py
   ```

6. **Classification with Random Forest**
   ```bash
   poetry run python random_forest.py
   ```

### Evaluating the Model
Check the output in the console for accuracy metrics and other performance indicators. Detailed reports will be saved in the specified directories within the project structure.

## Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Authors
- **jpotter80** - *Initial work* - [MyGithubProfile](https://github.com/jpotter80)

## Acknowledgments
- Vizier
    - This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France (DOI: 10.26093/cds/vizier). The original description of the VizieR service was published in A&AS 143, 23
- The Hipparcos and Tycho Catalogues (ESA 1997)
```

### Key Updates:
- **Database Setup**: Instructions added to set up the PostgreSQL database.
- **Poetry for Dependency Management**: Updated installation steps to use Poetry, enhancing project setup and dependency management.
- **Enhanced File Descriptions**: Updated to reflect the integration with PostgreSQL and the use of `database.py`.
