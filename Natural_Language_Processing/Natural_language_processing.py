# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:56:35 2018

@author: Shivam-PC
"""

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
y = dataset.iloc[:, 1].values# importing labels

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()# stemming 
stopwords = set(stopwords.words('english'))# importing stopwords
corpus = []
for i in range(0,len(dataset)):
    # removing punctuation and numbers and replaced by space
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])#don't remove a-z and A_Z
    # converting all letter to lower case
    review = review.lower()
    # Removing non-significant (Stopwords)
    review = review.split() #Spliting according to space
    review = [ps.stem(word) for word in review if not word in stopwords]
    #joining back the list
    review = ' '.join(review)
    corpus.append(review)

# Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus).toarray()

#spitting the dataset into training set and text set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)

# Fitting Naive Bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting Random Forest Classifier to training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

# Fitting Decision Tress Calssification to training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

from sklearn.ensemble import  AdaBoostClassifier
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train) 
print(classifier.score(X_test, y_test))

# predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)







