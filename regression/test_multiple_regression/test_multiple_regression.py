"""
    THIS CODE IS USED AS EDUCATION PURPOSE TO UNDERSTAND HOW LINEAR REGRESSION WORKS.
"""


def main():
    pass


def split_data(feature, target):
    from sklearn.model_selection import train_test_split
    feature_train, feature_test, target_train, target_test = train_test_split(feature,
                                                                              target,
                                                                              test_size=0.2,  # split with 80% train
                                                                              # and 20% test
                                                                              random_state=1)
    return feature_train, feature_test, target_train, target_test


if __name__ == '__main__':
    main()
