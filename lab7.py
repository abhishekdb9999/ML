import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore")

def simple_linear_regression():
    boston = fetch_openml(name='boston', version=1)
    X = boston.data
    y = boston.target
    X = X['RM'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual data')
    plt.plot(X_test, y_pred, color='red', label='Linear Regression line')
    plt.xlabel('Average number of rooms (RM)')
    plt.ylabel('House Price (MEDV)')
    plt.title('Linear Regression - Boston Housing')
    plt.legend()
    plt.show()

def simple_polynomial_regression():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ['mpg', 'cyl', 'disp', 'hp', 'weight', 'acc', 'year', 'origin', 'name']
    data = pd.read_csv(url, names=cols, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
    print(f"Missing values in the dataset before cleaning: {data.isnull().sum()}")
    data.dropna(subset=['hp', 'mpg'], inplace=True)
    if len(data) == 0:
        raise ValueError("No data available after dropping missing values in 'hp' or 'mpg'. Please check the dataset.")
    X = data[['hp']].values
    y = data['mpg'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual data')
    plt.plot(X_line, y_line, color='red', label='Polynomial Regression curve (degree 2)')
    plt.xlabel('Horsepower (hp)')
    plt.ylabel('Miles per Gallon (mpg)')
    plt.title('Polynomial Regression - Auto MPG')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Running Linear Regression for Boston Housing dataset...")
    simple_linear_regression()
    print("Running Polynomial Regression for Auto MPG dataset...")
    simple_polynomial_regression()
