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
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Installation
Clone the repository and navigate to the project directory. Install the required Python packages using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### File Descriptions
- `load.py`: Loads data from the CSV file.
- `clean.py`: Cleans and preprocesses the data.
- `explore.py`: Performs exploratory data analysis on the dataset.
- `visualize.py`: Generates visual plots to explore the data visually.
- `save.py`: Saves the cleaned data for further use.
- `clustering_analysis.py`: Performs clustering to find patterns in the data.
- `random_forest.py`: Implements the Random Forest classifier for star classification.

## Usage

### Running the Scripts
To run the scripts, navigate to the project directory and execute the following commands:

1. **Loading and cleaning data**
   ```bash
   python load.py
   python clean.py
   ```

2. **Exploratory Data Analysis**
   ```bash
   python explore.py
   ```

3. **Data Visualization**
   ```bash
   python visualize.py
   ```

4. **Clustering Analysis**
   ```bash
   python clustering_analysis.py
   ```

5. **Classification with Random Forest**
   ```bash
   python random_forest.py
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
	- This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France (DOI: 10.26093/cds/vizier). The original description of 	the VizieR service was published in A&AS 143, 23
- The Hipparcos and Tycho Catalogues (ESA 1997)
```