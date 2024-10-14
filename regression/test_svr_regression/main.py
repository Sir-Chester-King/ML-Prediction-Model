"""
    THIS CODE IS USED AS EDUCATION PURPOSE TO UNDERSTAND HOW LINEAR REGRESSION WORKS.
"""


def main():
    import pandas
    dataset = pandas.read_csv('Position_Salaries.csv')
    feature = dataset.iloc[:, 1:-1].values  # take only the second column 'cause it's the only one usefull to
    # predict the salary

    target = dataset.iloc[:, -1].values

    # --- FEATURE SCALING --- #
    """
    Applying the feature scaling to standardize the value used to train the model.
    """


if __name__ == '__main__':
    main()
