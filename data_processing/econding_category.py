# ---- IMPORT SECTION ---- #
import numpy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

"""
One-hot encoding (or "hot coding") is a technique used to convert categorical variables into a
format that can be fed to machine learning models.
Since machine learning models generally work best with numeric dataset, one-hot encoding transforms
categorical variables into numeric columns.
"""


def hot_coding_variables_feature(variables_feature):
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
                                                 remainder='passthrough')  # Using passthrough to keep the not
    # specified columns in the dataset.

    """
        Application the transformation ( using fit_transform() )and return of columns transformed as a NumPy array.
    """
    variables_feature = numpy.array(colum_transform_instance.fit_transform(variables_feature))

    return variables_feature


def hot_coding_variables_target(variables_target):
    """
        Sklearn LabelEncoder's class is used to transform categorical variables into numeric values.
        It is a useful tool when working with machine learning models that require numeric input, such as
        many classification and regression models.
    """

    """
        It apply an automatic coding, transforming the categorical values into a numeric values.
        The coding is for the 'Purchased' column that has only two value (yes and no).
    """
    label_encoder_vector = LabelEncoder()

    """
        It's a binary vector, so it doesn't necessary to cast it as a NON binary vector.    
    """
    variables_target = label_encoder_vector.fit_transform(variables_target)

    return variables_target
