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


data_FF = pd.read_sql_query(''' SELECT APINumber AS APINO, TotalBaseWaterVolume, CountyName, CountyNumber  FROM FracFocusRegistry WHERE (Statenumber = 49 AND (CountyName = 'Campbell' OR CountyName = 'Converse'));''', conn2)
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

# Number of unique API - After merging the two databases on API
data.APINO.nunique() # This gives us 233 well records

## Date Manipulation - Convert the date column from string to datetime format
#data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True, errors='ignore')
##data['Date'] = pd.to_datetime(data['Date'], errors='ignore')
## filtering the date for production after 2005
#data = data[(data['Date'] > '2005-01-01')] 

# Checking if there is any NULL value in the dataset
data.isnull().sum() 
data.dropna(axis=0, how='any') # entire row with even a single NA value will be removed - Better option to filter data

# At this point we have 458,703 rows and 12 columns

# Column for Cumulative value, Groupby function can be understood as (Split, Apply Function and Combine)
# Also converting the numbers to float 
data['cum_oil'] = data.groupby(['APINO'])['Oil'].apply(lambda x: x.cumsum()).astype(float)
data['cum_gas'] = data.groupby(['APINO'])['Gas'].apply(lambda x: x.cumsum()).astype(float)
data['cum_water'] = data.groupby(['APINO'])['Water'].apply(lambda x: x.cumsum()).astype(float)
data['cum_days'] = data.groupby(['APINO'])['Days'].apply(lambda x: x.cumsum()).astype(float)

# Sorting the table by APINO
data = data.sort_values(['APINO'])
data
# Now we need to add 30-60-90-180 and 365 day production 
# Let's just look into the oil for now!

#df = data[['APINO', 'Date', 'Oil', 'cum_oil', 'cum_days']].astype(int)
df = data[['APINO', 'Oil', 'cum_oil', 'Days', 'cum_days']].astype(float) #Removed the 'Date' column,  need to have a separate dataframe with datetime like values


df = df.reset_index()
df.index
df = df.sort_values(['index'])
df.to_csv(os.path.join(path,r'Data_Reduced.csv'))

# -----------------------------------------------------------

# Need a loop to change the dataframe name iteratively
time = 60
df['60_Interpol_OIL'] = 0.0
df['60_Interpol_OIL'].astype(str).astype(float)
for count in range(len(df['cum_oil'])):
    if (df['cum_days'][count] <= time and df['cum_days'][count+1] > time):
        df['60_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(time - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])   
pd.to_numeric(df['60_Interpol_OIL'], errors='coerce')
df['60_Interpol_OIL'] = df['60_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['60_Interpol_OIL'] != '0.0']

time = 180
df['180_Interpol_OIL'] = 0.0
df['180_Interpol_OIL'].astype(str).astype(float)
for count in range(len(df['cum_oil'])):
    if count>=1:
        if (df['cum_days'][count] <= time and df['cum_days'][count+1] > time):
            df['180_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(time - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])   
pd.to_numeric(df['180_Interpol_OIL'], errors='coerce')
df['180_Interpol_OIL'] = df['180_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['180_Interpol_OIL'] != '0.0']

time = 365
df['365_Interpol_OIL'] = 0.0
df['365_Interpol_OIL'].astype(str).astype(float)
for count in range(len(df['cum_oil'])):
    if count>=1:
        if (df['cum_days'][count] <= time and df['cum_days'][count+1] > time):
            df['365_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(time - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])   
pd.to_numeric(df['365_Interpol_OIL'], errors='coerce')
df['365_Interpol_OIL'] = df['365_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['365_Interpol_OIL'] != '0.0']

#-------------------------------------------------------------
  
# rename the column if they have the same name for 30,60,90,180,365 Interpolated values, if required
df.rename(columns={'60_Interpol_OIL': '60_day_cum_oil', '180_Interpol_OIL': '180_day_cum_oil', '365_Interpol_OIL': '365_day_cum_oil' }, inplace=True)

# import statsmodels and run basic analysis on 30,60,90, 180 and 365 data

import statsmodels.formula.api as smf
# from sklearn.linear_model import LinearRegression 
# scikitLearn is a Machine Learning Library in Python

# extracting only the relevant columns required for this point onwards
data_stats = df[['APINO', '60_day_cum_oil', '180_day_cum_oil', '365_day_cum_oil']].astype(float)
# fitting the linear regression model

# Model for 30-365 comparison  - Method 1 (Using Statsmodels)
X = data_stats['60_day_cum_oil']
Y = data_stats['365_day_cum_oil']
model1 = smf.ols(formula = 'Y ~ X', data=data_stats).fit()
print (model1.params)
print (model1.summary())

# Model for 60-365 comparison - Method 2 (ScikitLearn Methods)
X1 = data_stats[['180_day_cum_oil']]
Y1 = data_stats[['365_day_cum_oil']]
model2 = smf.ols(formula = 'Y ~ X', data=data_stats).fit()
print (model2.params)


from scipy.stats import norm
from scipy import stats
# Histograms and Density plots for all the columns calculated 
# Checking for the following statistical parameters
# 1. Normality - checking for the normal distribution
# 2. Homoscedasticity - assumption that dependent variables exhibit equal levels of variance across the range of predictor variables
# 3. Linearity - Good idea to check in case any data transformation is required
# 4. Absence of correlated errors

sns.distplot(data_stats['60_day_cum_oil'],fit = norm, hist=True, kde=True, bins=300, color = 'blue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
# Histogram (for kurtosis and skewness) and Normal Probability Plot (data distribution should follow the diagonal)
# Getting the value of Kurtosis and Skewness
print("Skewness: %f" % data_stats['365_day_cum_oil'].skew())
print("Kurtosis: %f" % data_stats['365_day_cum_oil'].kurt())

fig = plt.figure()
res = stats.probplot(data_stats['365_day_cum_oil'], plot=plt)
# Adding the labels
plt.title('Density Plot and Histogram of Annual Production')
plt.xlabel('Time')
plt.ylabel('Annual Production Frequency')

# Pairtplot - Useful for exploring correlations between multidimensional data
sns.pairplot(data_stats, size=5);

# Correlation Matrix and Heatmap
corr_matrix = data_stats.corr()
f, ax = plt.subplots(figsize = (6, 6))
sns.heatmap(corr_matrix, vmax=0.8, square=True)

# Jointplot - Useful for joint distribution between different datasets
#sns.jointplot("30_day_cum_oil", "365_day_cum_oil", data=data_stats, kind='reg');
sns.jointplot("60_day_cum_oil", "365_day_cum_oil", data=data_stats, kind='reg');
#sns.jointplot("90_day_cum_oil", "365_day_cum_oil", data_stats, kind='reg');
sns.jointplot("180_day_cum_oil", "365_day_cum_oil", data_stats, kind='reg');

# Convert the three dataframes we created to .csv file for tableau
data.to_csv(os.path.join(path,r'Data_Final.csv'))

cur2.close()
conn2.close()

cur.close()
conn.close()