#Sheri Johnson
#0205PythonDataPrepApplyAlzheimersML.py
#***I got sklearn installed!
#2/5 Python Alzheimers csv using apply
#read csv files, combine files on common column, create new columns w/ numerical for data analysis,
#delete text columns, save data to new csv file

#***This worked*** to install sklearn
#sudo apt-get install python3-sklearn python3-sklearn-lib python-sklearn-doc

import pandas as pd

#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

#physical activity level,depression level, Air Pollution , Income Level,
#stress level, 
def lowmedhigh(x): 
    if(x== 'Low'):
        return 0
    elif (x== 'Medium'):
        return 1
    else: #high
        return 2

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

 #   Diabetes, Hypertension, Family History of Alzheimers, Genetic Risk Factor
# Alzheimer’s Diagnosis
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

 #Employment Status
def employstatus(x):
    if(x == 'Unemployed'):
        return 0
    else: #Employed
        return 1
    
 #urban rural,    
def urbanrural(x):
    if(x == 'Urban'):
        return 0
    else: #Rural
        return 1
    
#Marital Status
def maritalstatus(x): 
    if(x== 'Single'):
        return 0
    elif (x== 'Divorced'):
        return 1
    else: #widowed
        return 2


#using apply to call the function and replace the text with numbers
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
df['Air Pollution Exposure'] = df['Air Pollution Exposure'].apply(lowmedhigh)
df['Employment Status'] = df['Employment Status'].apply(employstatus)
df['Genetic Risk Factor'] = df['Genetic Risk Factor'].apply(noyes)
df['Income Level'] = df['Income Level'].apply(lowmedhigh)
df['Stress Levels'] = df['Stress Levels'].apply(lowmedhigh)
df['Urban vs Rural Living'] = df['Urban vs Rural Living'].apply(urbanrural)
df["Alzheimer’s Diagnosis"] = df["Alzheimer’s Diagnosis"].apply(noyes)
df['Marital Status'] = df['Marital Status'].apply(maritalstatus)
df['Social Engagement Level'] = df['Social Engagement Level'].apply(lowmedhigh)
#print(df['Smoking Status'].head())

#drop country? use country code? or put country codes in country column?  ***There was header that had a special character-before we deleted it in the data file. 
#Do we need to add it in?  Family History of Alzheimerâ€™s
df.drop('Country', axis = 1, inplace= True)

#print(df.head())

#save to new file
df.to_csv('/home/sjohnson/datafile/alzheimers_prediction_updated_applytest.csv', index = False)

#print(df.isnull().sum())

#******NEW*******

#Copy the "Alzheimer’s Diagnosis" to compare to the prediction
y=df["Alzheimer’s Diagnosis"].copy()   
#Drop the "Alzheimer’s Diagnosis"column
df.drop(["Alzheimer’s Diagnosis"], axis=1, inplace=True)# axis = 1 indicates GRADE column  #accuracy .6475

#playing with other columns to drop to see if we can increase the accuracy
#df.drop(["Income Level"], axis=1, inplace=True) #no diff
df.drop(["Country Code"], axis=1, inplace=True) #up to .6549
df.drop(["Family History of Alzheimer’s"], axis=1, inplace=True) #.6518
df.drop(["Air Pollution Exposure"], axis=1, inplace=True) #.6536
df.drop(["Marital Status"], axis=1, inplace=True) #.6559  #.657 if with all the abov
df.drop(["Depression Level"], axis=1, inplace=True) #.658
#df.drop(["Genetic Risk Factor"], axis=1, inplace=True) #goes down
#df.drop(["Diabetes"], axis=1, inplace=True)#.655
df.drop(["Social Engagement Level"], axis=1, inplace=True) #.6596
#df.drop(["Stress Levels"], axis=1, inplace=True) #.657
#df.drop(["Physical Activity Level"], axis=1, inplace=True)#.656



X = df
'''
print(f"y values:\n{y}")
print(f"df values:\n{df}")
print(f"X values:\n{X}")
'''
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #.2 is 20% of the data for testing

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3) #check 3 neighbors, hard to figure out what the optimal # to use is

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test) #uses the trained model to predict the species for the test data
print(f"y_pred values:\n{y_pred}") #will produce list with same number of rows....using 0,1,2
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) #based on the data, here's how you're doing-if the number is realy low we 
#have to figure out what the problem is
print(f"Accuracy: {accuracy}") #print the accuracy on the console