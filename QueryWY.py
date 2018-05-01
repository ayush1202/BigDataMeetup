# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:44:38 2018

@author: ayushrastogi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 09:36:34 2018

@author: ayush
"""

import sqlite3                    
import pandas as pd
import numpy as np       
import matplotlib.pyplot as plt    
plt.style.use('ggplot')
#plt.rcdefaults()


conn = sqlite3.connect('C:/Users/ayush/OneDrive/ML DS Big Data/Meetup/Big Data Meetup/Wyoming Production Data/ProductiontableWY.db')
cur = conn.cursor()

#Entire data from database converted to a dataframe
data= pd.read_sql_query(" SELECT DISTINCT * FROM ProductiontableWY ORDER BY [Cum BOE] DESC LIMIT 100;", conn)
#print(data)

data.hist('Operator Alias')

#Data for Powder River Basin with a focus on Oil
data_powder = pd.read_sql_query(''' 
                                  SELECT DISTINCT 
                                  [API/UWI], [Operator Alias], [Reservoir], [Production Type],
                                  [Cum Gas], [Cum Oil], [Cum BOE],
                                  [Field], [Basin],[County/Parish]                         
                                  FROM ProductiontableWY
                                  WHERE ([Basin] = 'POWDER RIVER' AND [Production Type]='GAS')
                                  ORDER BY [Cum BOE] DESC
                                  LIMIT 100
                                  ;''', conn)

# print top 100 powder river dataset

plt.figure()
y = data_powder['Cum Gas'].values
x_name = data_powder['Operator Alias'].values
x = np.arange(len(data_powder['Operator Alias']))
plt.bar(x,y)
plt.xticks(x, x_name, rotation = 90)
plt.show()

#x = np.arange(len(data_powder['County/Parish']))
#plt.pie(x)
#plt.show()

x2 = data_powder['Cum BOE']
y2 = data_powder['Cum Gas']
plt.scatter(x2,y2)
plt.show()



