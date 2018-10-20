# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:11:40 2018

@author: Shivam-PC
"""

# importing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data from file
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori with the data
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)
for i in results:
   print(i[0])