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
filename = os.path.join(direc, 'titanic_train.csv')

data = pd.read_csv(filename)

median_age = data['age'].median()

data['age'].fillna(median_age, inplace=True)

data_inputs = data[['pclass', 'sex']]
expected_output = data[['survived']]

data_inputs['pclass'].replace({ '3rd' : 3, '2nd' : 2, '1st' : 1 }, inplace=True)
data_inputs['sex'].replace({ 'female': 0, 'male': 1 }, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(data_inputs, expected_output, test_size = 0.33, random_state=42)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

accuracy = rf.score(x_test, y_test)

joblib.dump(rf, "titanic_model2", compress=9)