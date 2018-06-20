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
path = r'C:\Users\ayush\Desktop\Meetup2_All Files'
os.chdir(path)

# Setting up the connections
conn = sqlite3.connect(r'C:\Users\ayush\Desktop\Meetup2_All Files\WY_Production.sqlite')
cur = conn.cursor()

#  Merge the data with the fracfocus database
conn2 = sqlite3.connect(r'C:\Users\ayush\Desktop\Meetup2_All Files\FracFocus.sqlite')
cur2 = conn2.cursor()
# Connections to the two databases complete

# SQL Query - Data from database converted to a dataframe
data = pd.read_sql_query(''' SELECT * FROM Production;''', conn) # Campbell County wells
data2 = pd.read_csv(r'C:\Users\ayush\Desktop\Meetup2_All Files\Converse.csv') #Converse County wells

# Append the production files
data_prod = data.append(data2)
print (data_prod.tail(10)) #default head() function prints 5 results
print (data_prod.shape) # in the form of rows x columns
data_prod.columns
data_prod.index
data_prod.describe() # get basic statistics for the dataset, does not include any string type elements
# Number of unique API 
data_prod.APINO.nunique() # This gives us 29,737 well records

# SL Query 2 - Import the data from FracFocus Database while selecting Campbell and Converse counties in Wyoming 
data_FF1 = pd.read_sql_query(''' SELECT APINumber AS APINO, TotalBaseWaterVolume, CountyName, CountyNumber  
                            FROM FracFocusRegistry 
                            WHERE (Statenumber = 49 AND (CountyNumber = 5 OR CountyNumber = 9));''', conn2)

data_FF2 = pd.read_sql_query(''' SELECT APINumber AS APINO, TotalBaseWaterVolume, CountyName, CountyNumber  
                            FROM registryupload
                            WHERE (Statenumber = 49 AND (CountyNumber = 5 OR CountyNumber = 9));''', conn2)

data_FF = pd.merge(data_FF1,data_FF2, on = 'APINO')

# Looking into the FracFocus database

print (data_FF.head(10)) #default head() function prints 5 results
print (data_FF.shape) # in the form of rows x columns
data_FF.columns
data_FF.index
data_FF.APINO.nunique() # This gives us 712 well records

# Look into the format in which APINumber is included in the Fracfocus Registry
data_FF['APINO'] 

# API Number Format Manipulation
data_prod['StateCode'] = '4900'
data_prod['Trail_zero'] = '0000'
data_prod['APINO'] = data_prod['StateCode'] + data_prod['APINO'].astype(str) + data_prod['Trail_zero']
data_prod['APINO']
data_prod.APINO.nunique() # This gives us 29,737 well records

# -------------Merging Dataframes (Merge the dataframes based on same API)
data_merged = pd.merge(data_prod,data_FF, on = 'APINO') # Default merge is on 'inner'
data_merged.APINO.nunique() # At this point we have 685 wells

## ------Date Manipulation - Convert the date column from string to datetime format
#data = data.drop(data[data.Days == 99].index) #multiple rows where days = 99 (incorrect default value)
#data.isnull().sum()  # Checking if there is any NULL value in the dataset
#data = data.dropna(axis=0, how='any') # entire row with even a single NA value will be removed - Better option to filter data


# Column for Cumulative value, Groupby function can be understood as (Split, Apply Function and Combine)
# Also converting the numbers to float 
data_merged['cum_oil'] = data_merged.groupby(['APINO'])['Oil'].apply(lambda x: x.cumsum()).astype(float)
data_merged['cum_gas'] = data_merged.groupby(['APINO'])['Gas'].apply(lambda x: x.cumsum()).astype(float)
data_merged['cum_water'] = data_merged.groupby(['APINO'])['Water'].apply(lambda x: x.cumsum()).astype(float)
data_merged['cum_days'] = data_merged.groupby(['APINO'])['Days'].apply(lambda x: x.cumsum()).astype(float)
# Another method for calculating cumulative sum based on a group
#data['cum_oil2'] = data.groupby('APINO')['Oil'].transform(pd.Series.cumsum)

# Sorting the table by APINO
data_merged = data_merged.sort_values(['APINO'])

# Let's just look into the oil for now!
data_merged.columns # the list of columns in the dataframe
df = data_merged[['APINO', 'cum_oil', 'cum_gas', 'cum_water', 'cum_days']].astype(float) # New Dataframe with selected columns 
df = df[(df[['cum_oil','cum_gas','cum_days']] != 0).all(axis=1)]
df

df = df.reset_index()
df.index
df = df.sort_values(['index'])
df.to_csv(os.path.join(path,r'Cumulative_Production_0619.csv')) # Converting the file to csv
df.APINO.nunique() # We have 682 wells left 


# -------------------------INTERPOLATION-------------------------------
# **Interpolation carried out on another script**

df = pd.read_csv(path+r'\Cumulative_Production_OGW_0619.csv')
df

#-------------Brief Statistical Analysis and Visualization - Oil Production------------------------------------------------

df.rename(columns={'60_Interpol_OIL': '60_day_cum_oil', '90_Interpol_OIL': '90_day_cum_oil', '180_Interpol_OIL': '180_day_cum_oil', '365_Interpol_OIL': '365_day_cum_oil', '730_Interpol_OIL': '730_day_cum_oil' }, inplace=True)
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
plt.subplot(211)
plt.scatter(X1, Y, marker='o', cmap='Dark2', color='r')
plt.title('Scatter Plot')
plt.xlabel("60 Day Production")
plt.ylabel ("365 Day Production")
plt.subplot(212)
plt.scatter(X2, Y, marker='.', cmap='Dark2',color='g')
plt.xlabel("180 Day Production")
plt.ylabel ("365 Day Production")

# Basic Statistical Analysis Using Statsmodels
model1 = smf.ols(formula = 'Y ~ X1', data=df).fit()
print (model1.summary())

# Method 2
model2 = smf.ols(formula = 'Y ~ X2', data=df).fit()
print (model2.params)
# print (model2.summary)

from scipy import stats
from scipy.stats import norm
# Histograms and Density plots for all the columns calculated 
# Check for the following statistical parameters
# 1. Normality - checking for the normal distribution
# 2. Homoscedasticity - assumption that dependent variables exhibit equal levels of variance across the range of predictor variables
# 3. Linearity - Good idea to check in case any data transformation is required
# 4. Absence of correlated errors

# Distplot - Visualizing the distribution of a dataset
# Histogram with KDE (Kernel Density Estimation is a non-parametric way to estimate the probability density function of a random variable)
sns.set()
sns.distplot(df['60_day_cum_oil'], axlabel=False, hist=True, kde=True, bins=50, color = 'blue',label ='60 Day Cumulative', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
sns.distplot(df['180_day_cum_oil'], axlabel=False, hist=True, kde=True, bins=50, color = 'red',label ='180 Day Cumulative', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
sns.distplot(df['365_day_cum_oil'], axlabel=False, hist=True, kde=True, bins=50, color = 'green',label ='365 Day Cumulative', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4}).set_title('Distribution Comparison')
plt.legend()
plt.show()

#skewness and kurtosis
print("Skewness: %f" % df['365_day_cum_oil'].skew())
print("Kurtosis: %f" % df['365_day_cum_oil'].kurt())


#Probability Plot - Quantiles - To get an idea about normality and see where the samples deviate from normality
sns.distplot(df['365_day_cum_oil'], fit = norm)
fig = plt.figure()
res = stats.probplot(df['365_day_cum_oil'], plot=plt)
# Adding the labels

# applying log transformation, in case of positive skewness, log transformations usually works well
#df['365_day_cum_oil_trans'] = np.log(df['365_day_cum_oil'])

# Pairtplot - Useful for exploring correlations between multidimensional data
sns.set(style="ticks", color_codes=True)
sns.pairplot(df, size=3, palette="husl", vars=["60_day_cum_oil", "180_day_cum_oil", "365_day_cum_oil"], kind="reg", markers=".")

# Correlation Matrix and Heatmap
df2 = df[["60_day_cum_oil", "90_day_cum_oil", "180_day_cum_oil","365_day_cum_oil","730_day_cum_oil"]]
corr_matrix = df2.corr()
f, ax = plt.subplots(figsize = (6, 6))
cm = sns.light_palette("green", as_cmap=True)
s = sns.heatmap(corr_matrix, vmax=0.8, square=True, annot=True, fmt=".2f", cmap = cm)

# Jointplot - Useful for joint distribution between different datasets
sns.jointplot(X1, Y, data=df, kind='reg')
sns.jointplot(X2, Y, data=df, kind='reg')

# Convert the three dataframes we created to .csv file for tableau
df.to_csv(os.path.join(path,r'Data_Final0619.csv'))

cur2.close()
conn2.close()

cur.close()
conn.close()