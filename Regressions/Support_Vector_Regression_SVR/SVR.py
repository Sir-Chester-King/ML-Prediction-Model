"""
    THIS CODE IS USED AS EDUCATION PURPOSE TO UNDERSTAND HOW REGRESSION WORKS.
"""


def main():
    import pandas
    dataset = pandas.read_csv('Position_Salaries.csv')
    feature = dataset.iloc[:, 1:-1].values  # take only the second column 'cause it's the only one usefull to
    # predict the salary

    target = dataset.iloc[:, -1].values

    print("-" * 40)
    print("-" * 40)
    print("'Features", end='\n')
    print(feature)

    print("-" * 40)
    print("-" * 40)
    print("'Target 1D Array", end='\n')
    print(target)

    # --- TRANSFORM TARGET ARRAY 1D IN AN ARRAY 2D, 'CAUSE THE STANDARD CLASS EXPECT A 2D ARRAY AS INPUT --- #
    target = target.reshape(len(target), 1)  # as argument the number of rows and the number of columns
    print("-" * 40)
    print("-" * 40)
    print("'Target 2D Array", end='\n')
    print(target)

    # --- FEATURE SCALING --- #
    """
    Applying the feature scaling to standardize the value used to train the model.
    Scale both feature and target 'cause the scale differences are huge, and to avoid the model
    use values out of scale, need to scale both, else the prediction will be not efficiency. 
    """
    from sklearn.preprocessing import StandardScaler
    sc_feature = StandardScaler()
    sc_target = StandardScaler()

    feature = sc_feature.fit_transform(feature)
    print("-" * 40)
    print("-" * 40)
    print("'Feature 2D Array SCALED", end='\n')
    print(feature)

    target = sc_target.fit_transform(target)
    print("-" * 40)
    print("-" * 40)
    print("'Target 2D Array SCALED", end='\n')
    print(target)

    # --- TRAIN SVR MODEL --- #
    from sklearn.svm import SVR
    svr_regression = SVR(kernel='rbf')  # Radial basis function kernel
    svr_regression.fit(feature, target)

if __name__ == '__main__':
    main()
