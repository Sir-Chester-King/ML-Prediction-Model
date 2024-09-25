"""
Class StandardScaler is used to standardise the feature of dataset.
"""

# ---- IMPORT SECTION ---- #
from sklearn.preprocessing import StandardScaler


def scaling_FEATURE_variables(feature_train, feature_test):
    sc = StandardScaler()
    feature_train[:, 3:] = sc.fit_transform(feature_train[:, 3:])  # Train the model with the data scaled
    feature_test[:, 3:] = sc.transform(feature_test[:, 3:])  # Apply the transformation to the new data

    return feature_train, feature_test
