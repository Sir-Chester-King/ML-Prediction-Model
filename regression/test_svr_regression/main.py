"""
    THIS CODE IS USED AS EDUCATION PURPOSE TO UNDERSTAND HOW LINEAR REGRESSION WORKS.
"""


def main():
    import pandas
    dataset = pandas.read_csv('50_Startups.csv')
    feature = dataset.iloc[:, :-1].values
    target = dataset.iloc[:, -1].values

    # --- ENCODING CATEGORICAL DATA --- #
    import numpy
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    column_transform_instance = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                                                  remainder='passthrough')
    feature = numpy.array(column_transform_instance.fit_transform(feature))

    # --- SPLIT DATA --- #
    from sklearn.model_selection import train_test_split
    feature_train, feature_test, target_train, target_test = train_test_split(feature,
                                                                              target, test_size=0.2, random_state=1)

    # --- MULTIPLE LINEAR REGRESSION --- #
    from sklearn.linear_model import LinearRegression
    mean_feature = numpy.mean(feature)
    mean_target = numpy.mean(target)

    multiple_lr_model = LinearRegression()  # Create instance of LinearRegression

    '''
        - Train the model to calculate the straight of linear regression using:
            1) X1...Xn = feature
            2) Y = target
        - .fit() method calculate the best optimal values for the  linear straight:
            B0 = Intercept
            B1....Bn = Coefficients
        - To calculate the optimal parameters of linear straight, the .fit() method use a
        Ordinary Least Squares (OLS) to calculate them.
    '''
    multiple_lr_model.fit(feature_train, target_train)
    b0 = multiple_lr_model.intercept_
    bx = multiple_lr_model.coef_
    print("-" * 40)
    print("-" * 40)
    print("Intercept: ", b0)
    print("Coefficients: ", bx)

    target_test_prediction = multiple_lr_model.predict(feature_test)
    print("-" * 40)
    print("-" * 40)
    print("TARGET TEST")
    print(target_test)

    print("-" * 40)
    print("-" * 40)
    print("PREDICTION TEST")
    print(target_test_prediction)

    numpy.set_printoptions(precision=2)

    print("-" * 40)
    print("-" * 40)
    print("CONCAT VECTORS - TARGET TEST & PREDICTION TEST")

    # Here the concat of 2 Numpy array:
    # 1: The array of the TARGET TEST
    # 2: The array of PREDICTION TEST
    # Using the Reshape function to a better view of the print, from horizontal to vertical
    vector_concat = numpy.concatenate(
        (target_test.reshape(len(target_test), 1),
         target_test_prediction.reshape(len(target_test_prediction), 1)
         ), 1)
    print(vector_concat)


if __name__ == '__main__':
    main()
