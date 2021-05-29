import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Data Preprocessing part
# loading dataset into x and y
dataset = pd.read_csv('insurance.csv')
# loading dataset into X independent variable and y dependent variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# there are categorical data in my dataset insurance.csv. In order to make a prediction we need to convert that
# data into something predictable. ColumnTransformer function makes it numeric.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
# then store that numeric data in my X variable.
X = np.array(ct.fit_transform(X))

dataset = pd.read_csv('WorldPopulation.csv')
X_1 = dataset.iloc[:, :-1].values
y_1 = dataset.iloc[:, -1].values


# Regression Part

# This method makes polynomial regression prediction. It takes two array, X independent variable and y dependent
# variable. It does not return anything it prints the r2_score that this prediction has.
def polynomial_regressor(X, y):
    # splitting dataset into train and test. We take 20 percent of the train data to test our result.
    # Random state is zero, if it's not it takes that 20 percent of data randomly, so in each execution result might
    # be different and that could cause inconsistency.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Calls polynomial regression with the degree of four.
    poly_reg = PolynomialFeatures(degree=4)
    # Transforming our train data to make polynomial regression.
    X_poly = poly_reg.fit_transform(X_train)
    # Since all regressions created upon linear regression. See report for details.
    regressor = LinearRegression()
    # fits our polynomial data and train data.
    regressor.fit(X_poly, y_train)
    # making prediction based on polynomial.
    y_pred = regressor.predict(poly_reg.transform(X_test))
    # making our print prettier, although we don't print any array at the moment.
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))
    # printing r2_score based on test and prediction values. Our performance increases when this score goes to 1.
    print(r2_score(y_test, y_pred))


# This method makes random forest regression prediction. It takes two array, X independent variable and y dependent
# variable. It does not return anything it prints the r2_score that this prediction has.
def random_forest_regression(X, y):
    # splitting dataset into train and test. We take 20 percent of the train data to test our result.
    # Random state is zero, if it's not it takes that 20 percent of data randomly, so in each execution result might
    # be different and that could cause inconsistency.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Calls random forest regression with the estimation of 10. For details about random forest regression, see report.
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    # Fits our random forest data and train data.
    regressor.fit(X_train, y_train)
    # making prediction based on X_test.
    y_pred = regressor.predict(X_test)
    # making our print prettier, although we don't print any array at the moment.
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))
    # printing r2_score based on test and prediction values. Our performance increases when this score goes to 1.
    print(r2_score(y_test, y_pred))


# This method makes multiple linear regression. It takes two array, X independent variable and y dependent variable.
# It does not return anything, just prints the r2_score that this prediction has.
def multiple_linear_regression(X, y):
    # splitting dataset into train and test. We take 20 percent of the train data to test our result.
    # Random state is zero, if it's not it takes that 20 percent of data randomly, so in each execution result might
    # be different and that could cause inconsistency.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Calls linear regression, but when we give more than one data in X. This class knows that we want to do multiple
    # linear regression.
    regressor = LinearRegression()
    # Fits our multiple regression data and train data.
    regressor.fit(X_train, y_train)
    # making prediction based on X_test
    y_pred = regressor.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))
    # printing r2_score based on test and prediction values. Our performance increases when this score goes to 1.
    print(r2_score(y_test, y_pred))


# This method makes support vector regression. It take two array, X independent variable and y dependent variable.
# It does not return anything, just prints the r2_score that this prediction has.
def support_vector_regression(X, y):
    # Reshaping y to make it 2d array. This line triggers a warning in console. Found out it was libraries version
    # issues and etc. Since it does not broke anything we decided to keep it.
    y = y.reshape(len(y), 1)
    # splitting dataset into train and test. We take 20 percent of the train data to test our result.
    # Random state is zero, if it's not it takes that 20 percent of data randomly, so in each execution result might
    # be different and that could cause inconsistency.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature scaling for X and Y. Support vector regression does work without scaling but it's not very good.
    # So feature scaling is important for this method.
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    # Using our scales we change X and y train to more precise results.
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)

    # Specifying which regression to use.
    regressor = SVR(kernel='rbf')
    # Fits our x and y train which we scaled.
    regressor.fit(X_train, y_train)
    # Making prediction. Note that we needed to use scale and give it a X_test in order to make this prediction.
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))
    # printing r2_score based on test and prediction values. Our performance increases when this score goes to 1.
    print(r2_score(y_test, y_pred))


# This method makes decision tree regression. It takes two array, X independent variable and y dependant variable.
# It does not return anything, just prints the r2_score that this prediction has.
def decision_tree_regression(X, y):
    # splitting dataset into train and test. We take 20 percent of the train data to test our result.
    # Random state is zero, if it's not it takes that 20 percent of data randomly, so in each execution result might
    # be different and that could cause inconsistency.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Using decision tree regressor.
    regressor = DecisionTreeRegressor(random_state=0)
    # Fits our X and y train into decision tree.
    regressor.fit(X_train, y_train)
    # Making prediction.
    y_pred = regressor.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1))
    # printing r2_score based on test and prediction values. Our performance increases when this score goes to 1.
    print(r2_score(y_test, y_pred))


# Calling regressions.

polynomial_regressor(X, y)
random_forest_regression(X, y)
multiple_linear_regression(X, y)
support_vector_regression(X, y)
decision_tree_regression(X, y)

decision_tree_regression(X_1, y_1)
polynomial_regressor(X_1, y_1)
random_forest_regression(X_1, y_1)
multiple_linear_regression(X_1, y_1)
support_vector_regression(X_1, y_1)

# https://www.kaggle.com/mirichoi0218/insurance
# https://www.kaggle.com/harlfoxem/housesalesprediction
# https://www.kaggle.com/dileep070/heart-disease-prediction-using-logistic-regression
# https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
# https://www.kaggle.com/sansuthi/world-population-by-year
# https://www.kaggle.com/andonians/random-linear-regression?select=train.csv
