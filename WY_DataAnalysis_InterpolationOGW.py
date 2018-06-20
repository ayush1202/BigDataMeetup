# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:33:01 2018

@author: AyushRastogi
"""

# Extracting the cumulative 60, 90, 180, 365 and 730 day production for Oil, Gas and Water

import pandas as pd
import os

os.getcwd() # Get the default working directory
path = r'C:\Users\ayush\Desktop\Meetup2_All Files'
os.chdir(path)

# Reading the input file with cumulative oil/gas/water production and cum days 
df = pd.read_csv(path+r'\Cumulative_Production_0619.csv')

# The process is repeated for Oil, Gas and Water

# --------------------------OIL Production------------------

# Creating the columns and filling them with 0
df['60_Interpol_OIL'] = 0
df['90_Interpol_OIL'] = 0
df['180_Interpol_OIL'] = 0
df['365_Interpol_OIL'] = 0
df['730_Interpol_OIL'] = 0

# For loop which runs through every row (until last but 1). If the cum_days value we need (60/90/180/365/730) fall in between the cell value, it uses linear
# interpolation and calculates the cumulative sum for that particular value

# y = y1 + ((y2-y1)*(x-x1)/(x2-x1)), where y = required production value, and x = 365 (Example)

for count in range(len(df['APINO'])-1): #loop running through the entire column
    if (df['cum_days'][count] < 60 and df['cum_days'][count+1] > 60):
            df['60_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(60 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 60): # if the required value is already present, simply copy it
        df['60_Interpol_OIL'][count] = df['cum_oil'][count]
pd.to_numeric(df['60_Interpol_OIL'], errors='coerce') # Convert the column values to numbers 
df['60_Interpol_OIL'] = df['60_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist() # Getting only 1 decimal place and adding values to a list
df[df['60_Interpol_OIL'] != '0.0'] # Getting rid of all the values which = 0.0
df['60_Interpol_OIL'].astype(float) # Convert the datatype to float (better since its a calculation)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 90 and df['cum_days'][count+1] > 90):
            df['90_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(90 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 90):
        df['90_Interpol_OIL'][count] = df['cum_oil'][count]
pd.to_numeric(df['90_Interpol_OIL'], errors='coerce')
df['90_Interpol_OIL'] = df['90_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['90_Interpol_OIL'] != '0.0']
df['90_Interpol_OIL'].astype(float)

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

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 730 and df['cum_days'][count+1] > 730):
            df['730_Interpol_OIL'][count] = df['cum_oil'][count-1] + ((df['cum_oil'][count+1]) - df['cum_oil'][count-1])*(730 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 730):
        df['730_Interpol_OIL'][count] = df['cum_oil'][count]
pd.to_numeric(df['730_Interpol_OIL'], errors='coerce')
df['730_Interpol_OIL'] = df['730_Interpol_OIL'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['730_Interpol_OIL'] != '0.0']
df['730_Interpol_OIL'].astype(float)


# --------------------------GAS Production------------------
df['60_Interpol_GAS'] = 0
df['90_Interpol_GAS'] = 0
df['180_Interpol_GAS'] = 0
df['365_Interpol_GAS'] = 0
df['730_Interpol_GAS'] = 0

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 60 and df['cum_days'][count+1] > 60):
            df['60_Interpol_GAS'][count] = df['cum_gas'][count-1] + ((df['cum_gas'][count+1]) - df['cum_gas'][count-1])*(60 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 60):
        df['60_Interpol_GAS'][count] = df['cum_gas'][count]
pd.to_numeric(df['60_Interpol_GAS'], errors='coerce')
df['60_Interpol_GAS'] = df['60_Interpol_GAS'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['60_Interpol_GAS'] != '0.0']
df['60_Interpol_GAS'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 90 and df['cum_days'][count+1] > 90):
            df['90_Interpol_GAS'][count] = df['cum_gas'][count-1] + ((df['cum_gas'][count+1]) - df['cum_gas'][count-1])*(90 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 90):
        df['90_Interpol_GAS'][count] = df['cum_gas'][count]
pd.to_numeric(df['90_Interpol_GAS'], errors='coerce')
df['90_Interpol_GAS'] = df['90_Interpol_GAS'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['90_Interpol_GAS'] != '0.0']
df['90_Interpol_GAS'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 180 and df['cum_days'][count+1] > 180):
            df['180_Interpol_GAS'][count] = df['cum_gas'][count-1] + ((df['cum_gas'][count+1]) - df['cum_gas'][count-1])*(180 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 180):
        df['180_Interpol_GAS'][count] = df['cum_gas'][count]
pd.to_numeric(df['180_Interpol_GAS'], errors='coerce')
df['180_Interpol_GAS'] = df['180_Interpol_GAS'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['180_Interpol_GAS'] != '0.0']
df['180_Interpol_GAS'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 365 and df['cum_days'][count+1] > 365):
            df['365_Interpol_GAS'][count] = df['cum_gas'][count-1] + ((df['cum_gas'][count+1]) - df['cum_gas'][count-1])*(365 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 365):
        df['365_Interpol_GAS'][count] = df['cum_gas'][count]
pd.to_numeric(df['365_Interpol_GAS'], errors='coerce')
df['365_Interpol_GAS'] = df['365_Interpol_GAS'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['365_Interpol_GAS'] != '0.0']
df['365_Interpol_GAS'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 730 and df['cum_days'][count+1] > 730):
            df['730_Interpol_GAS'][count] = df['cum_gas'][count-1] + ((df['cum_gas'][count+1]) - df['cum_gas'][count-1])*(730 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 730):
        df['730_Interpol_GAS'][count] = df['cum_gas'][count]
pd.to_numeric(df['730_Interpol_GAS'], errors='coerce')
df['730_Interpol_GAS'] = df['730_Interpol_GAS'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['730_Interpol_GAS'] != '0.0']
df['730_Interpol_GAS'].astype(float)

# ---------------------------Water Production-------------------

df['60_Interpol_WATER'] = 0
df['90_Interpol_WATER'] = 0
df['180_Interpol_WATER'] = 0
df['365_Interpol_WATER'] = 0
df['730_Interpol_WATER'] = 0

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 60 and df['cum_days'][count+1] > 60):
            df['60_Interpol_WATER'][count] = df['cum_water'][count-1] + ((df['cum_water'][count+1]) - df['cum_water'][count-1])*(60 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 60):
        df['60_Interpol_WATER'][count] = df['cum_water'][count]
pd.to_numeric(df['60_Interpol_WATER'], errors='coerce')
df['60_Interpol_WATER'] = df['60_Interpol_WATER'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['60_Interpol_WATER'] != '0.0']
df['60_Interpol_WATER'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 90 and df['cum_days'][count+1] > 90):
            df['90_Interpol_WATER'][count] = df['cum_water'][count-1] + ((df['cum_water'][count+1]) - df['cum_water'][count-1])*(90 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 90):
        df['90_Interpol_WATER'][count] = df['cum_water'][count]
pd.to_numeric(df['90_Interpol_WATER'], errors='coerce')
df['90_Interpol_WATER'] = df['90_Interpol_WATER'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['90_Interpol_WATER'] != '0.0']
df['90_Interpol_WATER'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 180 and df['cum_days'][count+1] > 180):
            df['180_Interpol_WATER'][count] = df['cum_water'][count-1] + ((df['cum_water'][count+1]) - df['cum_water'][count-1])*(180 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 180):
        df['180_Interpol_WATER'][count] = df['cum_water'][count]
pd.to_numeric(df['180_Interpol_WATER'], errors='coerce')
df['180_Interpol_WATER'] = df['180_Interpol_WATER'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['180_Interpol_WATER'] != '0.0']
df['180_Interpol_WATER'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 365 and df['cum_days'][count+1] > 365):
            df['365_Interpol_WATER'][count] = df['cum_water'][count-1] + ((df['cum_water'][count+1]) - df['cum_water'][count-1])*(365 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 365):
        df['365_Interpol_WATER'][count] = df['cum_water'][count]
pd.to_numeric(df['365_Interpol_WATER'], errors='coerce')
df['365_Interpol_WATER'] = df['365_Interpol_WATER'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['365_Interpol_WATER'] != '0.0']
df['365_Interpol_WATER'].astype(float)

for count in range(len(df['APINO'])-1):
    if (df['cum_days'][count] < 730 and df['cum_days'][count+1] > 730):
            df['730_Interpol_WATER'][count] = df['cum_water'][count-1] + ((df['cum_water'][count+1]) - df['cum_water'][count-1])*(730 - df['cum_days'][count-1])/(df['cum_days'][count+1]-df['cum_days'][count-1])
    elif (df['cum_days'][count] == 730):
        df['730_Interpol_WATER'][count] = df['cum_water'][count]
pd.to_numeric(df['730_Interpol_WATER'], errors='coerce')
df['730_Interpol_WATER'] = df['730_Interpol_WATER'].apply(lambda x: '%.1f' % x).values.tolist()
df[df['730_Interpol_WATER'] != '0.0']
df['730_Interpol_WATER'].astype(float)

# -----------------------------------------------------------------------

df.columns # Getting the list of columns
df.index # Getting the index values
# Grouping the values by the API Number and adding them based on the same unique index (API)
df = df.groupby(['APINO'])["60_Interpol_OIL", "90_Interpol_OIL", "180_Interpol_OIL", "365_Interpol_OIL", "730_Interpol_OIL", "60_Interpol_GAS", "90_Interpol_GAS", "180_Interpol_GAS", "365_Interpol_GAS", "730_Interpol_GAS", "60_Interpol_WATER", "90_Interpol_WATER", "180_Interpol_WATER", "365_Interpol_WATER","730_Interpol_WATER" ].apply(lambda x : x.astype(float).sum()).reset_index()
# Convert the file to .csv to use it with any data visualization tools
df.to_csv(os.path.join(path,r'Cumulative_Production_OGW_0619.csv'))
# -----------------------------------------------------------------------
