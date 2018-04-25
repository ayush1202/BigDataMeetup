# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:50:18 2018

@author: ayushrastogi
"""
#Create a database if the data is in the form of an .xlsx (Excel) file
import sqlite3
import pandas as pd
#Name of xlsx file. SQLite database will have the same name and extension .db
#Change the path to the production file

filename=r"C:\Users\ayush\OneDrive\ML DS Big Data\Meetup\Big Data Meetup\CO Production Data\excel\colorado"
con=sqlite3.connect(filename+"_excel_database.db")

#Using pandas to read the excel and function 'to_sql' to export the data
wb=pd.read_excel(filename+'.xlsx',sheetname=None)
for sheet in wb:
    wb[sheet].to_sql(sheet,con, index=False)
con.commit() #Committing the change
con.close() #Close the connection
