# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:33:01 2018

@author: AyushRastogi
"""

# Extracting the cumulative 60, 180, 365 day production for Gas, Oil and Water

import pandas as pd
import os

os.getcwd() # Get the default working directory
path = r'C:\Users\AyushRastogi\OneDrive\Meetup2\Ayush_Meetup2'
os.chdir(path)


df = pd.read_csv(path+r'\Cumulative_Production.csv')

# --------------------------OIL Production------------------
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



# --------------------------GAS Production------------------
df['60_Interpol_GAS'] = 0
df['180_Interpol_GAS'] = 0
df['365_Interpol_GAS'] = 0

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


# ---------------------------Water Production-------------------

df['60_Interpol_WATER'] = 0
df['180_Interpol_WATER'] = 0
df['365_Interpol_WATER'] = 0

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

# -----------------------------------------------------------------------
# Removing the 0's from all columns
df.columns
df.index
df = df.groupby(['APINO'])["60_Interpol_OIL", "180_Interpol_OIL", "365_Interpol_OIL", "60_Interpol_GAS", "180_Interpol_GAS", "365_Interpol_GAS","60_Interpol_WATER", "180_Interpol_WATER", "365_Interpol_WATER" ].apply(lambda x : x.astype(float).sum()).reset_index()
df.to_csv(os.path.join(path,r'Cumulative_Production_OGW.csv'))
# -----------------------------------------------------------------------

