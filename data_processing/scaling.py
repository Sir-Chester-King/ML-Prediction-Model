# ---- IMPORT SECTION ---- #
from sklearn.preprocessing import StandardScaler


def scaling_FEATURE_variables(feature_train, feature_test):
    sc = StandardScaler()
    feature_train[:, 3:] = sc.fit_transform(feature_train[:, 3:])
    feature_test[:, 3:] = sc.transform(feature_test[:, 3:])

    return feature_train, feature_test
