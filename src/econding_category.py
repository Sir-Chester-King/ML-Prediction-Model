# ---- IMPORT SECTION ---- #
import numpy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

"""
One-hot encoding (or "hot coding") is a technique used to convert categorical variables into a
format that can be fed to machine learning models.
Since machine learning models generally work best with numeric data, one-hot encoding transforms
categorical variables into numeric columns.
"""


def hot_coding(matrix_features_independent):
    """
        Sklearn ColumnTransformer's class, is a useful tool for applying different transformations
        to different columns in a dataset.
        It is especially useful when you are dealing with datasets that contain both numeric
        and categorical variables, and you want to apply different preprocessing techniques to
        each of them.
    """

    """
        The transformations to be applied to each group of columns are defined. 
        Each transformation is specified by a tuple containing:
        - name of transformation: encoder
        - Object of transformation: OneHotEncoder
        - Columns where apply the transformation: 'Country Column' from DataSet
        
    """
    colum_transform_instance = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                                                 remainder='passthrough')

    """
        Application the transformation ( using fit_transform() )and return of columns transformed as a NumPy array.
    """
    matrix_features_independent = numpy.array(colum_transform_instance.fit_transform(matrix_features_independent))

    return matrix_features_independent
