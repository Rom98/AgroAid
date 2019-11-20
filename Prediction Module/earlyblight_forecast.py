import pandas as pd
data = pd.read_csv('early_blight_data.csv')

from sklearn.preprocessing import LabelEncoder
lble = LabelEncoder()
data = data.apply(lble.fit_transform)
print(data)

x = data.iloc[:,0:2].values
y = data.iloc[:,2].values

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc = dtc.fit(x, y)
Y_pred = dtc.predict(x)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score: ", accuracy_score(y, Y_pred))

cnf = confusion_matrix(y,Y_pred)
print(cnf)

temp = int(input("Enter temp: "))
humidity = int(input("Enter humidity in percent: "))
wet_period = int(input("Enter wetness period: "))

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

t = [temp,wet_period]
print(t)

if (humidity<85):
    print("Likelihood is trivial (0)")
else:    
    user_input = pd.DataFrame({'Temperature': [t[0]], 'Wet_Period': [t[1]]})
    op = dtc.predict(user_input)
    print("Likelihood level is ",op)







