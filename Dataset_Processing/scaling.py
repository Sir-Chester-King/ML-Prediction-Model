"""
Class StandardScaler is used to standardise the feature of dataset.
Calculates the mean and standard deviation for each characteristic and uses them to transform the data, so that
the distribution of each feature has:
    - An average equal to 0,
    - A standard deviation equal to 1.
"""

# ---- IMPORT SECTION ---- #
from sklearn.preprocessing import StandardScaler


def scaling_FEATURE_variables(feature_train, feature_test):
    sc = StandardScaler()
    feature_train[:, 3:] = sc.fit_transform(feature_train[:, 3:])  # Train the model with the data scaled
    feature_test[:, 3:] = sc.transform(feature_test[:, 3:])  # Apply the transformation to the new data

    return feature_train, feature_test
