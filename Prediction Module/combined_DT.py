import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import sys

def encode_for_late(inp):
    #first arg = temperature in farenheit
    #second arg = humidity
    wet_period = inp[1]
    temp = inp[0]

    if temp<40:
        temp = 1        #Low
    if temp>=40 and temp<56:
        temp = 2        #Mild
    if temp>=56 and temp<86:
        temp = 3        #Moderate
    if temp>=86: 
        temp = 0        #High

    
    if wet_period<9:
        wet_period = 2                      #Small
    if wet_period>=9 and wet_period < 16:
        wet_period = 1                      #Medium
    if wet_period>=16 and wet_period < 20:
        wet_period = 0                      #Long
    if wet_period>=21:
        wet_period = 3                      #Very Long
    
    inp = [temp,wet_period]
    
    return inp

def encode_for_early(inp):
    
    temp = inp[0]
    wet_period = inp[1]
    
    if temp<55:
        temp = 1        #Low
    if temp>=55 and temp<80:
        temp = 2        #Mild
    if temp>=80 and temp<91:
        temp = 3        #Moderate
    if temp>=91: 
        temp = 0        #High
    
    
    if wet_period<4:
        wet_period = 4                      #Very Small
    if wet_period>=4 and wet_period < 9:
        wet_period = 2                      #MSmall
    if wet_period>=9 and wet_period < 16:
        wet_period = 1                      #Medium
    if wet_period>=16 and wet_period < 21:
        wet_period = 0                      #Long
    if wet_period>=21:
        wet_period = 3                      #Very Long
    inp = [temp,wet_period]
    
    return(inp)
    
input_arguments = [int(sys.argv[1]),int(sys.argv[3])]
humidity  = int(sys.argv[2])
s1 = 0
s2 = 0
if(humidity>=75):
    inp1 = encode_for_late(input_arguments)
    data = pd.read_csv('late_blight_data.csv')
    lble = LabelEncoder()
    data = data.apply(lble.fit_transform)
    #print(data)
    
    x = data.iloc[:,0:2].values
    y = data.iloc[:,2].values
    
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(x, y)
    Y_pred = dtc.predict([inp1])
    s1 = Y_pred[0]
    print (s1)

if(humidity>=85):
        
    inp2 = encode_for_early(input_arguments)
    data = pd.read_csv('early_blight_data.csv')
    lble = LabelEncoder()
    data = data.apply(lble.fit_transform)
    #print(data)
    
    x = data.iloc[:,0:2].values
    y = data.iloc[:,2].values
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(x, y)
    Y_pred = dtc.predict([inp2])
    s2 = Y_pred[0]
    print (s2)
#return (s1,s2)
    
