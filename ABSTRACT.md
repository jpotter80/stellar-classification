### Project Abstract: Stellar Classification Using Machine Learning

**Project Title**: Stellar Classification Using Machine Learning for Giant and Dwarf Star Differentiation

**Authors**: James Potter 

#### Abstract

This project focuses on classifying stars into giant and dwarf categories based on their spectral characteristics and other relevant features using machine learning techniques. The primary objective is to develop a robust model capable of accurately differentiating between giant and dwarf stars, leveraging a dataset containing stellar attributes such as visual magnitude (Vmag), parallax (Plx), parallax error (e_Plx), color index (B-V), and spectral type (SpType).

#### Introduction

Stellar classification is a fundamental task in astrophysics, aiding in the understanding of stellar evolution, galactic formation, and other cosmic phenomena. Traditional methods of classification rely heavily on manual inspection of spectral data. However, with the advent of machine learning, it is possible to automate and enhance the accuracy of stellar classification.

#### Methodology

1. **Data Collection and Preparation**:
   - The dataset, named `Star99999_raw.csv`, comprises 99,999 entries with features including `Vmag`, `Plx`, `e_Plx`, `B-V`, `SpType`, and the target variable `StarType`.
   - Initial data exploration revealed a significant number of missing values, particularly in the `SpType` and `StarType` columns.
   - A unique list of spectral types was generated to facilitate the creation of a binary classification scheme distinguishing between giant and dwarf stars.

2. **Data Cleaning and Imputation**:
   - The columns `Vmag`, `Plx`, `e_Plx`, and `B-V` were converted to numeric types.
   - Missing values in these numeric columns were imputed using the median value to ensure data integrity.
   - The `SpType` column was imputed with 'Unknown', and missing values in the `StarType` column were inferred using the cleaned data and initial classification results.

3. **Feature Engineering**:
   - A binary column `StarType` was created, categorizing stars as either 'Giant' or 'Dwarf' based on their spectral type and other relevant features.
   - The cleaned and imputed dataset was split into training and testing sets for model evaluation.

4. **Model Training and Evaluation**:
   - A RandomForestClassifier was selected for its robustness and ability to handle complex datasets.
   - Hyperparameter tuning was performed using GridSearchCV to optimize model performance.
   - The final model achieved an accuracy of 83.27%, with balanced precision, recall, and F1-scores for both giant and dwarf star classifications.

5. **Model Deployment**:
   - The trained model was deployed to predict star types on new data, with predictions saved to a CSV file (`predictions.csv`).
   - A PostgreSQL database was integrated into the project to store the cleaned data and predictions, showcasing SQL skills and enabling further analysis.

6. **Database Integration**:
   - The cleaned data and predictions were loaded into PostgreSQL tables named `cleaned_star_data` and `predicted_star_data`.
   - Column names were updated to ensure clarity and consistency.

#### Results

The machine learning model successfully classified stars with an accuracy of 83.27%. Key metrics include:
- Precision (Dwarf): 0.77
- Recall (Dwarf): 0.88
- F1-Score (Dwarf): 0.82
- Precision (Giant): 0.90
- Recall (Giant): 0.79
- F1-Score (Giant): 0.84

The database integration facilitated efficient data management, allowing for seamless storage and retrieval of both raw and predicted star classifications.

#### Conclusion

This project demonstrates the effectiveness of machine learning in automating stellar classification, providing a robust and scalable solution for differentiating between giant and dwarf stars. The integration of a PostgreSQL database enhances data accessibility and management, laying the foundation for future research and analysis in astrophysical studies. Future work may include refining the model with additional features, exploring other classification algorithms, and expanding the dataset to include more diverse stellar types.

#### Keywords

Stellar Classification, Machine Learning, Random Forest, Data Cleaning, Feature Engineering, PostgreSQL, Astrophysics
