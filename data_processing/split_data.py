# ---- IMPORT SECTION ---- #
from sklearn.model_selection import train_test_split


def split_DATASET(feature, target):
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2,
                                                                              random_state=1)
    return feature_train, feature_test, target_train, target_test
