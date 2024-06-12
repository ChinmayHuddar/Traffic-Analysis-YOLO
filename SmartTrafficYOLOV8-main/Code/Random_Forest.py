# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:54:28 2024

@author: ROG ZEPHYRUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import statistics as st
 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
 
warnings.filterwarnings('ignore')

df= pd.read_csv('SignalTiming.csv')
#print(df)



# Assuming df is your DataFrame
x = df.iloc[:, 0:3]  #features
y = df.iloc[:, 4:5]  # Target variable

#print(x)
#print(y)

regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
 
# Fit the regressor with x and y data
regressor.fit(x, y)

oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')
 
# Making predictions on the same data or new data
predictions = regressor.predict(x)
 
# Evaluating the model
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')
 
r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')

standard_deviation = np.std(y)
print(f'Standard-Deviaton: {standard_deviation}')

custom_MW = 10
custom_LMV = 20
custom_HMV = 2

# Create a DataFrame with the custom values
custom_data = pd.DataFrame({'MW': [custom_MW], 'LMV': [custom_LMV], 'HMV': [custom_HMV]})

# Make predictions using the trained model
predicted_value = regressor.predict(custom_data)

print("Predicted value:", predicted_value[0])