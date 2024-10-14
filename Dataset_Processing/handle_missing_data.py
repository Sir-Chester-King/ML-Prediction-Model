# ---- IMPORT SECTION ---- #
import numpy
from sklearn.impute import *

# ---- TAKE CARE OF MISSING DATA SECTION ---- #
"""
Function to handle the missing values in the dataset.
To perform that it be used the class SimpleImputer of SKLEARN library, used to handle missing dataset within a dataset.

In case on missing dataset, them will be replaced with the average of all value in the column.
The dataset replaced are only numbers, not strings.
"""


def handle_missing_data(variables_feature):
    """
        missing_values: Specifies the value to be considered missing.
        By default, it is set to numpy.nan, but can be set to other values if the missing dataset is
        represented differently.

        strategy: Defines the strategy for replacing missing values.
        Mean option: Replaces missing values with the average of the column (only for numeric variables).
    """
    imputer_istance = SimpleImputer(missing_values=numpy.nan, strategy='mean')  # Object IMPUTER.

    """
        To handle the missing dataset in the dataset it used the .fit() function.
        The .fit() method of Imputer classes is used to calculate and store the statistics needed to 
        fill in missing dataset in a dataset. 

        In other words, fit() trains the Imputer on the dataset, identifying the information 
        needed to handle NaN values.
    """

    """
    CALCULATE THE STATISTICS NEEDED TO FILL IN MISSING DATA IN DATASET.
        Colum 1 = Age
        Colum 2 = Salary
        This links the Imputer instance to the matrix of features.
    """

    imputer_istance.fit(variables_feature[:, 1:3])  # Specified all raw's and the first and second colum.

    """
    STORE THE PREVIOUSLY CALCULATED DATA IN THE MISSING DATA IN DATASET.
        Colum 1 = Age
        Colum 2 = Salary
    """
    # This method returns the new version od dataset (with the average of missing dataset).
    new_dataset = imputer_istance.transform(
        variables_feature[:, 1:3])  # Passed as argument the columns of the dataset.

    """
    UPDATING THE NEW VERSION OF DATASET IN THE OLD DATASET, TO HAVE A DATASET WITHOUT MISSING DATA.
    """
    variables_feature[:, 1:3] = new_dataset

    return variables_feature
