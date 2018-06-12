# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:45:51 2018

@author: AyushRastogi
"""
import sqlite3                    
import pandas as pd # data processing and csv file IO library
import numpy as np       
import matplotlib.pyplot as plt
import seaborn as sns # python graphing library
plt.style.use('seaborn')
sns.set(style="white", color_codes=True)
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

# plt.rcdefaults() # resest to default matplotlib parameters 
import warnings #ignore unwanted messages
warnings.filterwarnings("ignore")

conn = sqlite3.connect(r'C:\Users\AyushRastogi\OneDrive\Meetup2\Ayush_Meetup2\WY_Production.sqlite')
cur = conn.cursor()

# SQL Query - Entire data from database converted to a dataframe
data = pd.read_sql_query(''' SELECT * FROM Production;''', conn)
print (data.head(10)) #default head() function prints 5 results
print (data.shape) # in the form of rows x columns
data.columns
data.index
data.describe() # get basic statistics for the dataset, does not include any string type elements

# Date Manipulation - Convert the date column from string to datetime format
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True, errors='ignore')
#data['Date'] = pd.to_datetime(data['Date'], errors='ignore')
# filtering the date for production after 2005
data = data[(data['Date'] > '2005-01-01')] 
# Sorting the data based on APINO and then by Date
#data.sort_values(['APINO', 'Date'])
data.sort_values(['APINO'])
data

# Checking if there is any NULL value in the dataset
data.isnull().sum() 
data.dropna(axis=0, how='any') # entire row with even a single NA value will be removed - Better option to filter data

#Column for Cumulative value, Groupby as (Split, Apply Function and Combine)
# Also converting the numbers to float 
data['cum_oil'] = data.groupby(['APINO'])['Oil'].apply(lambda x: x.cumsum()).astype(int)
data['cum_gas'] = data.groupby(['APINO'])['Gas'].apply(lambda x: x.cumsum()).astype(int)
data['cum_water'] = data.groupby(['APINO'])['Water'].apply(lambda x: x.cumsum()).astype(int)
data['cum_days'] = data.groupby(['APINO'])['Days'].apply(lambda x: x.cumsum()).astype(int)
data

#Since a certain  API seems to be having very high, cum_days, we need to look into it specifically to make sure no errors
# Same API with different ID sequences, maybe produced by multiple operators at different instances
# data.ix[data['APINO']=='705008']

# Now we need to add 30-60-90-180 and 365 day production 
# Let's just look into the oil for now!

#df = data[['APINO', 'Date', 'Oil', 'cum_oil', 'cum_days']].astype(int)
df = data[['APINO', 'Oil', 'cum_oil', 'cum_days']].astype(int) #Removed the 'Date' column,  need to have a separate dataframe with datetime like values

df['Interpol_OIL'] = 0.0 #create a new column with 0 value
type(df)

df = df.reset_index()
df.index

# -----------------------------------------------------------

# will do the calculations for all 5 time periods, but all have same name as 'Interpol_OIL' - Need a method to change the dataframe name inside the loop
time = [60,90,180,365]
# will do the calculations for all 5 time periods, but all have same name as 'Interpol_OIL' - Need a method to change the dataframe name inside the loop
for time in time:
    for count in range(len(df['cum_oil'])):
        if (df['cum_days'][count] <= time and df['cum_days'][count+1] > time):
            df['Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(time - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])   
    
    pd.to_numeric(df['Interpol_OIL'], errors='coerce')
    #df['Interpol_OIL'] = df['Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['Interpol_OIL'] != '0.0']

# rename the column if they have the same name for 30,60,90,180,365 Interpolated values
df.rename(columns={'Interpol_Oil': '30_day_cum_oil', 'Interpol_Oil2': '60_day_cum_oil','Interpol_Oil3': '90_day_cum_oil', 'Interpol_Oil4': '180_day_cum_oil', 'Interpol_Oil5': '365_day_cum_oil' }, inplace=True)

# import statsmodels and run basic analysis on 30,60,90, 180 and 365 data

import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression 
# scikitLearn is a Machine Learning Library in Python

# extracting only the relevant columns required for this point onwards
data_stats = data[['30_day_cum_oil', '60_day_cum_oil', '90_day_cum_oil', '180_day_cum_oil', '365_day_cum_oil']].astype(int)
# fitting the regression model

# Model for 30-365 comparison  - Method 1 (Using Statsmodels)
X = data['30_day_cum_oil']
Y = data['365_day_cum_oil']
model1 = smf.ols(formula = 'Y ~ X', data=data_stats).fit()
print (model1.params)
print (model1.summary())

# Model for 60-365 comparison - Method 2 (ScikitLearn Methods)
X1 = data['60_day_cum_oil']
Y1 = data['365_day_cum_oil']
model2 = LinearRegression()
model2.fit(X1, Y1)
print (model2.coef_)
print (model2.intercept_)

# Model for 90-365 comparison - Method 2 (ScikitLearn Methods)
X1 = data['60_day_cum_oil']
Y1 = data['365_day_cum_oil']
model2 = LinearRegression()
model2.fit(X1, Y1)
print (model2.coef_)
print (model2.intercept_)

# Model for 180-365 comparison - Method 2 (ScikitLearn Methods)
X1 = data['60_day_cum_oil']
Y1 = data['365_day_cum_oil']
model2 = LinearRegression()
model2.fit(X1, Y1)
print (model2.coef_)
print (model2.intercept_)

# histograms and density plots for all the columns calculated 
# Crossplots and Linear regression on the calculated columns 
# Convert to csv file for tableau
