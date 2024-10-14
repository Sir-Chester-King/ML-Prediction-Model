"""
    THIS CODE IS USED AS EDUCATION PURPOSE TO UNDERSTAND HOW LINEAR REGRESSION WORKS.
"""


def main():
    import numpy
    import pandas
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    dataset = pandas.read_csv('Salary_Data.csv')
    feature = dataset.iloc[:, :-1].values  # Gather the columns of features (Independent variables)
    target = dataset.iloc[:, -1].values  # Gather the columns of targets (Dependent variables)

    mean_feature = numpy.mean(feature)
    mean_target = numpy.mean(target)

    # Split the dataset to use the 80% of data to a train model and
    # the 20% to use as a Test (new data we want to predict values)
    feature_train, feature_test, target_train, target_test = split_data(feature, target)

    print("-" * 40)
    print("-" * 40)
    print("'Feature' TRAIN SET", end='\n')
    print(feature_train)

    print("-" * 40)
    print("-" * 40)
    print("'Feature' TEST SET", end='\n')
    print(feature_test)

    print("-" * 40)
    print("-" * 40)
    print("'Target' TRAIN SET", end='\n')
    print(target_train)

    print("-" * 40)
    print("-" * 40)
    print("'Target' TEST SET", end='\n')
    print(target_test)

    regression = LinearRegression()  # Instance of Linear Regression class to calculate the straight line.

    '''
        - Train the model to calculate the straight of linear Regressions using:
            1) x = feature
            2) y = target
        - .fit() method calculate the best optimal values for the  linear straight:
            B0 = Intercept
            B1 = Coefficient
        - To calculate the optimal parameters of linear straight, the .fit() method use a
        Ordinary Least Squares (OLS) to calculate them.
    '''
    regression.fit(feature_train, target_train)
    b0 = regression.intercept_
    b1 = regression.coef_

    print("-" * 40)
    print("-" * 40)
    print("Intercept: ", b0)
    print("Coefficient: ", b1)

    # Now the "MODEL" is trained (after calculated B0 and B1), and we can "PREDICT" the values of new data.
    # Passed as parameter the new data we want to predict values
    target_test_prediction = regression.predict(feature_train)
    print("-" * 40)
    print("-" * 40)
    print("'Target' PREDICTION Linear Regression Straight", end='\n')
    print(target_test_prediction)

    def plot_TRAIN():
        # Creating the graph
        plt.figure(figsize=(10, 6))  # Defying the size of the graph

        # Creating the scatter plot with real data (feature_train and target_train)
        plt.scatter(feature_train, target_train, color='blue', label='Real Data')

        # Creating the linear Regressions straight
        plt.plot(feature_train, target_test_prediction, color='red', label='Regression Straight')

        # Defying label and title of plot
        plt.title('Scatter Plot - Dataset TRAINING', fontsize=14)
        plt.xlabel('Year of experience', fontsize=12)
        plt.ylabel('Salary', fontsize=12)

        # View the mean (average) of FEATURE and TARGET
        plt.axhline(y=mean_target, color='green', linestyle='--', label=f'Average Salary = {mean_target}')
        plt.axvline(x=mean_feature, color='orange', linestyle='--',
                    label=f'Average Year of experience = {mean_feature}')

        # Show Legend
        plt.legend()

        # Show plot
        plt.grid(True)
        plt.show()

    def plot_TEST():
        # Creating the scatter plot with real data (feature_train and target_train)
        plt.scatter(feature_test, target_test, color='blue', label='Real Data')

        # Creating the linear Regressions straight
        plt.plot(feature_train, target_test_prediction, color='red', label='Regression Straight')

        # Defying label and title of plot
        plt.title('Scatter Plot - Dataset TARGET', fontsize=14)
        plt.xlabel('Year of experience', fontsize=12)
        plt.ylabel('Salary', fontsize=12)

        # Show Legend
        plt.legend()

        # Show plot
        plt.grid(True)
        plt.show()

    plot_TRAIN()
    plot_TEST()


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

'''
# Function to calculate the linear Regressions without the class LinearRegression.
def calculate_linear_regression():
    import numpy
    # Average of Feature and Target
    mean_feature = numpy.mean(feature)
    mean_target = numpy.mean(target)

    print(mean_feature, mean_target)

    # Calculate the Coefficients of Regressions
    beta_1 = numpy.sum((feature - mean_feature) * (target - mean_target)) / numpy.sum((feature - mean_feature) ** 2)
    beta_0 = mean_target - beta_1 * mean_feature

    # Straight of Regressions
    target_prediction = beta_0 + beta_1 * feature
'''
