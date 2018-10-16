# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:06:04 2018

@author: Shivam-PC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

##spitting the dataset into training set and text set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)
#
##feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
# Fitting Polynimial Regressuin to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)# degree is highest power of x
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
print( lin_reg_2.predict(poly_reg.fit_transform(6.5)))
