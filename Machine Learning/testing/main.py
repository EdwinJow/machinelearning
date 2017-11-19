"""Testing some machine learning stuff"""
import os
import pandas as pd

direc = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(direc, 'ziphouseholddetail.csv'))

print(pd.crosstab(df['MedianHouseholdValue'], df['STAbv'], margins=True))