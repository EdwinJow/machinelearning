'''testing pandas'''
import os
import pandas as pd
import matplotlib.pyplot as plt

direc = os.path.dirname(__file__)
filename = os.path.join(direc, 'hubble_data.csv')

df = pd.read_csv(filename)
df.plot(x='distance', y='recession_velocity')
plt.show()
