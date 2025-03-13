#Sheri Johnson
#3/12 Python Alzheimers csv using apply, was originaly 0205DataPrepApply
#read csv files, combine files on common column, create new columns w/ numerical for data analysis,
#delete text columns, save data to new csv file

import pandas as pd
#import numpy as np

#read data
df = pd.read_csv('/home/sjohnson/datafile/alzheimers_prediction_dataset.csv')
#print(df.head())

df2 = pd.read_csv('/home/sjohnson/datafile/CountryCodes.csv')
#print(df2.head())

#add country code to df
df = df.join(df2.set_index('Country'), on=['Country'])

def convert_gender(Gender):
    if(Gender == 'Male'):
        return 0
    else: #Female
        return 1

#physical activity level,depression level 
def lowmedhigh(x): 
    if(x== 'Low'):
        return 0
    elif (x== 'Medium'):
        return 1
    else: #high
        return 2
'''   
def convert_smokingstatus(SmokingStatus): 
    if(SmokingStatus== 'Never'):
        return 0
    elif (SmokingStatus== 'Former'):
        return 1
    else: #Current
        return 2
'''
#smoking status
def neverformercurrent(x): 
    if(x== 'Never'):
        return 0
    elif (x== 'Former'):
        return 1
    else: #Current
        return 2
    
 #alcohol consumption
def neveroccasionally(x): 
    if(x== 'Never'):
        return 0
    elif (x== 'Occasionally'):
        return 1
    else: #regularly
        return 2

 #   Diabetes, Hypertension, Family History of Alzheimers
def noyes(x):
    if(x == 'No'):
        return 0
    else: #Yes
        return 1

 #cholesterol Level,    
def normalhigh(x):
    if(x == 'Normal'):
        return 0
    else: #High
        return 1

#Sleep quality
def pooravggood(x): 
    if(x== 'Poor'):
        return 0
    elif (x== 'Average'):
        return 1
    else: #good
        return 2
#Dietary Habits
def unhealthyavghealthy(x): 
    if(x== 'Unhealthy'):
        return 0
    elif (x== 'Average'):
        return 1
    else: #healthy
        return 2

df['Gender'] = df['Gender'].apply(convert_gender)
df['Physical Activity Level'] = df['Physical Activity Level'].apply(lowmedhigh)
df['Smoking Status'] = df['Smoking Status'].apply(neverformercurrent)
df['Alcohol Consumption'] = df['Alcohol Consumption'].apply(neveroccasionally)
df['Diabetes'] = df['Diabetes'].apply(noyes)
df['Hypertension'] = df['Hypertension'].apply(noyes)
df['Cholesterol Level'] = df['Cholesterol Level'].apply(normalhigh)
df["Family History of Alzheimer’s"] = df["Family History of Alzheimer’s"].apply(noyes)
df['Depression Level'] = df['Depression Level'].apply(lowmedhigh)
df['Sleep Quality'] = df['Sleep Quality'].apply(pooravggood)
df['Dietary Habits'] = df['Dietary Habits'].apply(unhealthyavghealthy)




print(df['Smoking Status'].head())