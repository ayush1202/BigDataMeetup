# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:07:42 2018

@author: ayush
"""

# -*- coding: utf-8 -*-

import sqlite3
import pandas

#Store the data through pandas in a dataframe
data=pandas.read_csv(r'C:/Users/ayushrastogi/OneDrive/ML DS Big Data/Meetup/Big Data Meetup/Wyoming Production Data/Production Table.csv')

#Creating the database
connection = sqlite3.connect('C:/Users/ayushrastogi/OneDrive/ML DS Big Data/Meetup/Big Data Meetup/Wyoming Production Data/ProductiontableWY.db')
cursor = connection.cursor()

#Create the table
def create_table():
    cursor.execute('DROP TABLE IF EXISTS ProductiontableWY')
    cursor.execute('''CREATE TABLE ProductionTableWY(
                   API INTEGER,
                   OperatorAlias TEXT,
                   LeaseName	TEXT,
                   WellNumber	TEXT,
                   EntityType	TEXT,
                   County TEXT,
                   Reservoir	TEXT,
                   ProductionType	TEXT,
                   ProducingStatus	TEXT,
                   DrillType	TEXT,
                   FirstProdDate	TIMESTAMP,
                   LastProdDate	TIMESTAMP,
                   CumGas	REAL,
                   CumOil	REAL,
                   CumBOE	REAL,
                   CumWater	REAL,
                   CumMMCFGE	REAL,
                   CumBCFGE	REAL,
                   DailyGas	REAL,
                   DailyOil	REAL,
                   FirstMonthOil	REAL,
                   FirstMonthGas	REAL,
                   FirstMonthWater	REAL,
                   First6Oil	REAL,
                   First6Gas	REAL,
                   First6BOE	REAL,
                   First6Water	REAL,
                   First12Oil	REAL,
                   First12Gas	REAL,
                   First12BOE	REAL,
                   First12MMCFGE	REAL,
                   First12Water	REAL,
                   First24Oil	REAL,
                   First24Gas	REAL,
                   First24BOE	REAL,
                   First24MMCFGE	REAL,
                   First24Water	REAL,
                   First60Oil	REAL,
                   First60Gas	REAL,
                   First60BOE	REAL,
                   First60Water	REAL,
                   First60MMCFGE	REAL,
                   PracIPOilDaily	REAL,
                   PracIPGasDaily	REAL,
                   PracIPCFGED	 REAL,
                   PracIPBOE	REAL,
                   LatestOil	REAL,
                   LatestGas	REAL,
                   LatestWater	REAL,
                   Prior12Oil	REAL,
                   Prior12Gas	REAL,
                   Prior12Water	REAL,
                   LastTestDate	TIMESTAMP,
                   LastFlowPressure	REAL,
                   LastWHSIP	REAL,
                   SecondMonthGOR	 REAL,
                   LatestGOR	REAL,
                   CumGOR	REAL,
                   Last12Yield	 REAL,
                   SecondMonthYield	REAL,
                   LatestYield	REAL,
                   PeakGas	REAL,
                   PeakGasMonthNo	REAL,
                   PeakOil	REAL,
                   PeakOilMonthNo	REAL,
                   PeakBOE	REAL,
                   PeakBOEMonthNo	REAL,
                   PeakMMCFGE	REAL,
                   PeakMMCFGEMonthNo	REAL,
                   UpperPerforation	REAL,
                   LowerPerforation	REAL,
                   GasGravity	REAL,
                   OilGravity	REAL,
                   CompletionDate	TIMESTAMP,
                   WellCount	INTEGER,
                   MaxActiveWells	REAL,
                   MonthsProduced	REAL,
                   GasGatherer	TEXT,
                   OilGatherer	TEXT,
                   LeaseNumber	REAL,
                   SpudDate	TIMESTAMP,
                   MeasuredDepthTD	REAL,
                   TrueVerticalDepth	REAL,
                   PerforatedIntervalLength	REAL,
                   Field	TEXT,
                   State	TEXT,
                   District	TEXT,
                   Basin	TEXT,
                   Country	TEXT,
                   Section	INTEGER,
                   Township	TEXT,
                   Range	TEXT,
                   Abstract	REAL,
                   Block	REAL,
                   Survey	REAL,
                   OCSArea	TEXT,
                   PGCArea TEXT,
                   SurfaceLatitudeWGS84	REAL,
                   SurfaceLongitudeWGS84	REAL,
                   ReportedOperator	TEXT,
                   Last12Oil	REAL,
                   Last12Gas	REAL,
                   Last12Water 	REAL,
                   EntityID	INTEGER);
                    ''')
connection.commit()

# Using pandas to export data from .csv to SQLite
df = pandas.read_csv('C:/Users/ayushrastogi/OneDrive/ML DS Big Data/Meetup/Big Data Meetup/Wyoming Production Data/Production Table.csv')
df.to_sql("ProductiontableWY", connection, if_exists='append', index=False)

# Close the csv file, commit changes, and close the connection
connection.commit()
connection.close()