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
data.ix[data['APINO']=='705008']

# Now we need to add 30-60-90-180 and 365 day production 
# Let's just look into the oil for now!

df = data[['APINO', 'Date', 'Oil', 'cum_oil', 'cum_days']]
df['Interpol_OIL'] = 0.0 #create a new column with 0 value
type(df)

df = df.reset_index()

time = [30,60,90,180,365]
# will do the calculations for all 5 time periods, but all have same name as 'Interpol_OIL' - Need a method to change the dataframe name inside the loop
for time in time:
    for count in range(len(df['cum_oil'])):
        if (df['cum_days'][count] <= 365 and df['cum_days'][count+1] > 365):
            df['Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(365 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])   
    
    pd.to_numeric(df['Interpol_OIL'], errors='coerce')
    df['Interpol_OIL'] = df['Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()

df[df['Interpol_OIL'] != '0.0']

# remove columns where days = 0
#columns = ['Days']
#data = data.replace(0, pd.np.nan).dropna(axis=0, how='any', subset=columns).fillna(0)
#data.sort_values(['APINO', 'Date'])

# Handling missing data
data.isnull().sum() # count number of null values in the dataset
data.dropna() # remove entire rows where NA values present
data.dropna(thresh=2) # keep only the rows with atleast 2 non-NA values
data.dropna(axis=1, how='all') # Only the columns with all NA values will be removed 
data.dropna(axis=1, how='any') # entire column with even a single NA value will be removed
data.dropna(axis=0, how='any') # entire row with even a single NA value will be removed - Better option to filter data

#data2 = pd.DataFrame([[1., 6.5, 3.], [1., np.NaN, np.NAN],
#                     [NAN, NAN, NAN], [NAN, 6.5, 3.]])
#data2
#data.fillna(0) # filling the NA values with 0
#data.fillna(data.mean()) # filling the average value with mean, useful in many cases where average parameter holds more importance
#data2.fillna(method='ffill') #here ffill is the 'forward fill'
