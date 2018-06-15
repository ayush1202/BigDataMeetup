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
import os

os.path
os.getcwd() # Get the default working directory
path = r'C:\Users\AyushRastogi\OneDrive\Meetup2\Ayush_Meetup2'
os.chdir(path)

# Setting up the connections
conn = sqlite3.connect(r'C:\Users\AyushRastogi\OneDrive\Meetup2\Ayush_Meetup2\WY_Production.sqlite')
cur = conn.cursor()

#  Merge the data with the fracfocus database
conn2 = sqlite3.connect(r'C:\Users\AyushRastogi\OneDrive\Meetup2\Meetup1\FracFocus.sqlite')
cur2 = conn2.cursor()
# Connections to the two databases complete

# SQL Query - Data from database converted to a dataframe
data = pd.read_sql_query(''' SELECT * FROM Production;''', conn)
print (data.head(10)) #default head() function prints 5 results
print (data.shape) # in the form of rows x columns
data.columns
data.index
data.describe() # get basic statistics for the dataset, does not include any string type elements
# Number of unique API 
data.APINO.nunique() # This gives us 27,742 well records

data_FF = pd.read_sql_query(''' SELECT APINumber AS APINO, TotalBaseWaterVolume, CountyName, CountyNumber  
                            FROM FracFocusRegistry 
                            WHERE (Statenumber = 49 AND (CountyName = 'Campbell' OR CountyName = 'Converse'));''', conn2)
print (data_FF.head(10)) #default head() function prints 5 results
print (data_FF.shape) # in the form of rows x columns
data_FF.columns
data_FF.index
data_FF.APINO.nunique() # This gives us 654 well records

# Look into the format in which APINumber is included in the Fracfocus Registry
data_FF['APINO'] 

# API Number Format Manipulation
data['StateCode'] = '4900'
data['Trail_zero'] = '0000'
data['APINO'] = data['StateCode'] + data['APINO'] + data['Trail_zero']
data['APINO']

# Merge the two dataframes based on same API
data = pd.merge(data,data_FF,on = 'APINO')

data = data.drop(data[data.Days == 99].index) #multiple rows where days = 99 (incorrect default value)
data.shape

# Number of unique API - After merging the two databases on API, Need to know since we lose some wells after merging tables
data.APINO.nunique() # This gives us 233 well records

## Date Manipulation - Convert the date column from string to datetime format
#data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True, errors='ignore')

# Checking if there is any NULL value in the dataset
data.isnull().sum() 
data = data.dropna(axis=0, how='any') # entire row with even a single NA value will be removed - Better option to filter data

data.shape
# At this point we have 385,852 rows and 12 columns, as well as 228 unique well records

# Column for Cumulative value, Groupby function can be understood as (Split, Apply Function and Combine)
# Also converting the numbers to float 
data['cum_oil'] = data.groupby(['APINO'])['Oil'].apply(lambda x: x.cumsum()).astype(float)
data['cum_gas'] = data.groupby(['APINO'])['Gas'].apply(lambda x: x.cumsum()).astype(float)
data['cum_water'] = data.groupby(['APINO'])['Water'].apply(lambda x: x.cumsum()).astype(float)
data['cum_days'] = data.groupby(['APINO'])['Days'].apply(lambda x: x.cumsum()).astype(float)
# Another method for calculating cumulative sum based on a group
#data['cum_oil2'] = data.groupby('APINO')['Oil'].transform(pd.Series.cumsum)

# Sorting the table by APINO
data = data.sort_values(['APINO'])
data

# Now we need to add 30-60-90-180 and 365 day production 
# Let's just look into the oil for now!

data.columns # the list of columns in the dataframe
# Only looking at oil production for analysis
df = data[['APINO', 'cum_oil', 'cum_days']].astype(float) # New Dataframe with selected columns 
df = df.reset_index()
df.index
df = df.sort_values(['index'])
df.to_csv(os.path.join(path,r'Data_Reduced.csv')) # Converting the file to csv

# There are certain wells which are producing from 60's so they have more than 40 years of production
# If required to filter those wells out, the following code chunk will help

## Date Manipulation - Convert the date column from string to datetime format
#data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True, errors='ignore')
##data['Date'] = pd.to_datetime(data['Date'], errors='ignore')
## filtering the date for production after 2005
#data = data[(data['Date'] > '2005-01-01')] 

# ----------------------------------------------------------------------
df = pd.read_csv(path+r'\Data_Reduced.csv')
df

df['60_Interpol_OIL'] = 0
df['180_Interpol_OIL'] = 0
df['365_Interpol_OIL'] = 0

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 60 and df['cum_days'][count+1] > 60):
            df['60_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(60 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 60):
        df['60_Interpol_OIL'][count] = df['cum_oil'][count]
pd.to_numeric(df['60_Interpol_OIL'], errors='coerce')
df['60_Interpol_OIL'] = df['60_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['60_Interpol_OIL'] != '0.0']
df['60_Interpol_OIL'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 180 and df['cum_days'][count+1] > 180):
            df['180_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(180 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 180):
        df['180_Interpol_OIL'][count] = df['cum_oil'][count]
pd.to_numeric(df['180_Interpol_OIL'], errors='coerce')
df['180_Interpol_OIL'] = df['180_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['180_Interpol_OIL'] != '0.0']
df['180_Interpol_OIL'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 365 and df['cum_days'][count+1] > 365):
            df['365_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(365 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 365):
        df['365_Interpol_OIL'][count] = df['cum_oil'][count]
pd.to_numeric(df['365_Interpol_OIL'], errors='coerce')
df['365_Interpol_OIL'] = df['365_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['365_Interpol_OIL'] != '0.0']
df['365_Interpol_OIL'].astype(float)

df = df.groupby(['APINO'])["60_Interpol_OIL", "180_Interpol_OIL", "365_Interpol_OIL"].apply(lambda x : x.astype(float).sum()).reset_index()
df


#-------------Brief Statistical Analysis and Visualization------------------------------------------------

df.rename(columns={'60_Interpol_OIL': '60_day_cum_oil', '180_Interpol_OIL': '180_day_cum_oil', '365_Interpol_OIL': '365_day_cum_oil' }, inplace=True)
df.columns

# import statsmodels and run basic analysis on 60,180 and 365 data

# Descriptive Statistics
df.describe()

df.set_index('APINO')
import statsmodels.formula.api as smf

# Scatter Plot
X1 = df['60_day_cum_oil']
X2 = df['180_day_cum_oil']
Y = df['365_day_cum_oil']
plt.scatter(X1, Y)

# Basic Statistical Analysis Using Statsmodels
model1 = smf.ols(formula = 'Y ~ X1', data=df).fit()
print (model1.summary())

# Method 2 (ScikitLearn Methods)
model2 = smf.ols(formula = 'Y ~ X2', data=df).fit()
print (model2.params)
# print (model2.summary)

from scipy import stats
from scipy.stats import norm
# Histograms and Density plots for all the columns calculated 
# Checking for the following statistical parameters
# 1. Normality - checking for the normal distribution
# 2. Homoscedasticity - assumption that dependent variables exhibit equal levels of variance across the range of predictor variables
# 3. Linearity - Good idea to check in case any data transformation is required
# 4. Absence of correlated errors

sns.set()
sns.distplot(df['60_day_cum_oil'], hist=True, kde=True, bins=50, color = 'blue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
sns.distplot(df['180_day_cum_oil'], hist=True, kde=True, bins=50, color = 'red', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
sns.distplot(df['365_day_cum_oil'], hist=True, kde=True, bins=50, color = 'green', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
sns.plt.legend()

#skewness and kurtosis
print("Skewness: %f" % df['365_day_cum_oil'].skew())
print("Kurtosis: %f" % df['365_day_cum_oil'].kurt())

sns.distplot(df['365_day_cum_oil'], fit = norm)
fig = plt.figure()
res = stats.probplot(df['365_day_cum_oil'], plot=plt)
# Adding the labels
plt.title('Density Plot and Histogram of Annual Production')

# applying log transformation, in case of positive skewness, log transformations usually works well
df['365_day_cum_oil_tr'] = np.log(df['365_day_cum_oil'])

sns.distplot(df['365_day_cum_oil_tr'], fit = norm)
fig = plt.figure()
res = stats.probplot(df['365_day_cum_oil_tr'], plot=plt)

# Pairtplot - Useful for exploring correlations between multidimensional data
sns.pairplot(df, size=3)

# Correlation Matrix and Heatmap
corr_matrix = df.corr()
f, ax = plt.subplots(figsize = (6, 6))
cm = sns.light_palette("green", as_cmap=True)
s = sns.heatmap(corr_matrix, vmax=0.8, square=True, annot=True, fmt=".2f", cmap = cm)

# Jointplot - Useful for joint distribution between different datasets
sns.jointplot(X1, Y, data=df, kind='reg');
sns.jointplot(X2, Y, data=df, kind='reg');

# Convert the three dataframes we created to .csv file for tableau
data.to_csv(os.path.join(path,r'Data_Final.csv'))

cur2.close()
conn2.close()

cur.close()
conn.close()