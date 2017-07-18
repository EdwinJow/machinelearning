'''tutorial from sentdex'''
import os
import quandl
import configparser
import datetime
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style

style.use('ggplot')

direc = os.path.abspath(os.curdir)
file_path = os.path.join(direc, 'config.ini')
parser = configparser.ConfigParser()
parser.read(file_path)

#grab api key from config file
quandl.ApiConfig.api_key = parser['api_keys']['quandl']

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#calculate some new metrics in new columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#change columns to only those that are needed
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

#replace NaN
df.fillna(-99999, inplace=True)

#days to forecast out
forecast_out = int(math.ceil(0.01*len(df)))
print('forecast out days: ', forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

#features
X = np.array(df.drop(['label'], 1))

#normalize features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#label
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

#split data into training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#n_jobs=-1 sets max processing power
clf = LinearRegression(n_jobs=-1)

#pass trianing data to algorithm
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

#get dates
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#print(accuracy)


