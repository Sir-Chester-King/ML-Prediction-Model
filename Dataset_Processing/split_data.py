"""
The Split Dataset is used to split the .csv file into the TRAIN data, used to train the model, and into
TEST data, used to predict the target data.
"""

# ---- IMPORT SECTION ---- #
from sklearn.model_selection import train_test_split


def split_DATASET(feature, target):
    # Split the data into TRAIN and TEST data.
    """
    Split:
    80% Train to train the model;
    20% Test used to test the model.
    """
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2,
                                                                              random_state=1)
    return feature_train, feature_test, target_train, target_test
