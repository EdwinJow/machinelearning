'''testing basic machine learning using titanic dataset'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#machine learning algorithm
from sklearn.ensemble import RandomForestClassifier

#functionality to split training and test data
from sklearn.model_selection import train_test_split

#write model to file
from sklearn.externals import joblib 

pd.options.mode.chained_assignment = None
direc = os.path.dirname(__file__)
filename = os.path.join(direc, 'titanic_test.csv')

data = pd.read_csv(filename)

#fill NaN age with median age
median_age = data['age'].median()
data['age'].fillna(median_age, inplace=True)

data_extract = data[['pclass', 'sex', 'age']]

#convert data to numeric for the training model
data_extract['pclass'].replace({ '3rd' : 3, '2nd' : 2, '1st' : 1 }, inplace=True)
data_extract['sex'].replace({ 'female': 0, 'male': 1 }, inplace=True)

data_extract_model2 = data_extract[['pclass', 'sex']]

#load model2 with training data (class and sex only)
rf = joblib.load(os.path.join(direc, 'titanic_model2'))

#predict who survives
pred = rf.predict(data_extract_model2)
print(pred)

#load model2 with training data
rf2 = joblib.load(os.path.join(direc, 'titanic_model1'))
#predict who survives
pred = rf2.predict(data_extract)
print(pred)
