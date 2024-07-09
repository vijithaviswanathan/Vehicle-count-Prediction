# importing the pandas module for 
# data frame
import pandas as pd
 
 
# load the data set into train variable.
train = pd.read_csv('vehicles.csv')
 
# display top 5 values of data set
train.head()
# function to get all data from time stamp
 
# get date
def get_dom(dt):
    return dt.day
 
# get week day
def get_weekday(dt):
    return dt.weekday()
 
# get hour
def get_hour(dt):
    return dt.hour
 
# get year
def get_year(dt):
    return dt.year
 
# get month
def get_month(dt):
    return dt.month
 
# get year day
def get_dayofyear(dt):
    return dt.dayofyear
 
# get year week
def get_weekofyear(dt):
    return dt.weekofyear
 
 
train['DateTime'] = train['DateTime'].map(pd.to_datetime)
train['date'] = train['DateTime'].map(get_dom)
train['weekday'] = train['DateTime'].map(get_weekday)
train['hour'] = train['DateTime'].map(get_hour)
train['month'] = train['DateTime'].map(get_month)
train['year'] = train['DateTime'].map(get_year)
train['dayofyear'] = train['DateTime'].map(get_dayofyear)
train['weekofyear'] = train['DateTime'].map(get_weekofyear)
 
# display
train.head()
# there is no use of DateTime module
# so remove it
train = train.drop(['DateTime'], axis=1)
 
# separating class label for training the data
train1 = train.drop(['Vehicles'], axis=1)
 
# class label is stored in target
target = train['Vehicles']
 
print(train1.head())
target.head()
#importing Random forest
from sklearn.ensemble import RandomForestRegressor
 
#defining the RandomForestRegressor
m1=RandomForestRegressor()
 
m1.fit(train1,target)
#testing
m1.predict([[11,6,0,1,2015,11,2]])
