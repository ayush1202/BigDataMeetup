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
plt.style.use('seaborn')
#plt.rcdefaults()


conn = sqlite3.connect('C:/Users/ayushrastogi/OneDrive/ML DS Big Data/Meetup/Big Data Meetup/Wyoming Production Data/ProductiontableWY.db')
cur = conn.cursor()

# SQL Query - Entire data from database converted to a dataframe
data= pd.read_sql_query(" SELECT * FROM ProductiontableWY ORDER BY [Cum BOE] DESC;", conn)
print(data.head(10))
print (data.shape)

index_data  = data.index
print(index_data)
col = data.columns
print(col)
data.shape

data_distinct = pd.read_sql_query(" SELECT DISTINCT * FROM ProductiontableWY ORDER BY [Cum BOE] DESC;", conn)
data_distinct.shape
# Total Rows 287,992
# Distinct Rows 143,996


# Plot 1: Most Productive Reservoirs (Cum BOE) in Entire Wyoming Dataset 
fig, ax = plt.subplots()
data_distinct['Operator Alias'].value_counts()[:10].plot(kind='barh', figsize = (10,10), )
plt.xlabel('Well Count', fontsize = 12, fontweight='bold')
plt.ylabel('Operators', fontsize = 12, fontweight='bold')
plt.title ('Top 10 Producing Operators(By Well Count)', fontsize = 14, fontweight='bold')
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid = True
plt.tight_layout()

data_distinct['Basin'].value_counts()[:5].plot(kind='barh', figsize = (10,10))
plt.xlabel('Well Count', fontsize = 12, fontweight='bold')
plt.ylabel('Basins', fontsize = 12, fontweight='bold')
plt.title('Top 5 Basins (By Well Count)', fontsize = 14, fontweight='bold')
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid = True
plt.tight_layout()
plt.show()

#Data for Powder River Basin with a focus on Oil
data_powder = pd.read_sql_query(''' 
                                  SELECT DISTINCT 
                                  [API/UWI], [Operator Alias], [Reservoir], [Production Type],
                                  [Cum Gas], [Cum Oil], [Cum BOE],
                                  [Field], [Basin],[County/Parish]                         
                                  FROM ProductiontableWY
                                  WHERE ([Basin] = 'POWDER RIVER' AND [Production Type]='OIL')
                                  ORDER BY [Cum Oil] DESC
                                  LIMIT 10
                                  ;''', conn)

# print top 100 powder river dataset with respect to Oil Production

#Plot 2: Identify the operator with highest oil production

#xmin = min(data_powder['Operator Alias'])
#ymin = min(data_powder['Cum BOE'].values)
#xmax = max(data_powder['Operator Alias'])
#ymax = max(data_powder['Cum BOE'].values)
#ax.set_xlim([xmin,xmax])
#axes.set_ylim([ymin,ymax])
fig, ax = plt.subplots()
axes = plt.gca()
x = np.arange(len(data_powder['Operator Alias']))
y = data_powder['Cum Oil'].values
x_name = data_powder['Operator Alias']
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
fig.subplots_adjust(top=0.85)
ax.set_title('Plot 1: Operator with Higest Production', fontsize=14, fontweight='bold')
ax.set_xlabel('Operator', fontsize = 12, fontweight='bold')
ax.set_ylabel('Cum BOE', fontsize = 12, fontweight='bold')
plt.grid = False
plt.xticks(x, x_name, rotation=90)
fig.set_size_inches(10, 10)
ax.bar(x,y, align='center', alpha=0.5, color = 'b')
plt.tight_layout()
plt.show()