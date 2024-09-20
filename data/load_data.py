# ---- IMPORT SECTION ---- #
import os

import dotenv
import pandas

from src import check_missing_value


# ---- LOAD DATA FROM CSV SECTION ---- #
def load_data():
    dotenv.load_dotenv()  # Load the .env file to gather the path of CSV file.
    dataset = pandas.read_csv(os.getenv('PATH_DATA'))  # Gather the dataset from the CSV file.

    # Check if there are missing values in the various columns (only numbers' value).
    has_nan = check_missing_value.check_missing_value(dataset.iloc[:, 1:3].values)

    # Gather all the values in the columns from the dataset, except the last one; as a numpy array.
    # Used the ILOC indexer in PANDAS to select the subset of the data and store that in the variable.
    matrix_features_independent = dataset.iloc[:, :-1].values

    # Gather all the values in the last column from the dataset; as a numpy array.
    # Used the ILOC indexer in PANDAS to select the subset of the data and store that in the variable.
    vector_features_dependent = dataset.iloc[:, -1].values

    return matrix_features_independent, vector_features_dependent, has_nan
