# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:45:51 2018

@author: AyushRastogi
"""

# This file is no longer useful since the command is merged into the main file 

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

conn = sqlite3.connect(r'C:\Users\AyushRastogi\Downloads\BigDataMeetup-master\PowderDb.sqlite')
cur = conn.cursor()

# SQL Query - Entire data from database converted to a dataframe
data = pd.read_sql_query(''' SELECT * FROM Production;''', conn)
print (data.head(10)) #default head() function prints 5 results
print (data.shape) # in the form of rows x columns

data.describe() # get basic statistics for the dataset, does not include any string type elements

# Handling missing data
data.isnull().sum() # count number of null values in the dataset
data.dropna() # remove entire rows where NA values present
data.dropna(thresh=2) # keep only the rows with atleast 2 non-NA values
data.dropna(axis=1, how='all') # Only the columns with all NA values will be removed 
data.dropna(axis=1, how='any') # entire column with even a single NA value will be removed
data.dropna(axis=0, how='any') # entire row with even a single NA value will be removed - Better option to filter data

data2 = pd.DataFrame([[1., 6.5, 3.], [1., np.NaN, np.NAN],
                     [NAN, NAN, NAN], [NAN, 6.5, 3.]])
data2
data.fillna(0) # filling the NA values with 0
data.fillna(data.mean()) # filling the average value with mean, useful in many cases where average parameter holds more importance
data2.fillna(method='ffill') #here ffill is the 'forward fill'
