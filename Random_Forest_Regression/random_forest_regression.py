# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:41:09 2018

@author: Shivam-PC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Fitting Random forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# Predicting a new result with Random forest regression
y_pred =regressor.predict(6.5)

# Visualising the Random forest regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random forest regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()