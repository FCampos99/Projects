# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
#matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
#read the data
train = pd.read_csv("train.csv", sep=",", header=0, decimal=".")

test = pd.read_csv("test.csv", sep=",", header=0, decimal=".")

#EXPLORING DATA 
#Fill age null values
train['Age'] = train['Age'].fillna(train['Age'].median())
#Separate men and women in different variables
women = train[train["Sex"] == "female"]
men = train[train["Sex"] == "male"]
womenS = train[(train['Sex'] == "female") & (train['Survived'] == 1)]
womenD = train[(train['Sex'] == "female") & (train['Survived'] == 0)]
menS = train[(train['Sex'] == "male") & (train['Survived'] == 1)]
menD = train[(train['Sex'] == "female") & (train['Survived'] == 0)]
#compare the death rates of both genders
train['Died'] = 1 - train['Survived']
train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(10, 5),  stacked=True)
#Compare the age gaps of dead people by gender
figure = plt.figure(figsize=(10, 5))
plt.hist([women[women['Survived'] == 1]['Age'], train[train['Survived'] == 0]['Age']], 
        stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of female passengers')
plt.legend();
figure = plt.figure(figsize=(10, 5))
plt.hist([men[men['Survived'] == 1]['Age'], men[men['Survived'] == 0]['Age']], 
        stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of male passengers')
plt.legend();

#Comparing the rate of dead people by the fare of the ticket
fig = plt.figure(figsize=(10, 5))
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], 
        stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();
plt.show()

#Comparing the rate of dead people by the class of the passenger
fig = plt.figure(figsize=(10, 5))
plt.hist([train[train['Survived'] == 1]['Pclass'], train[train['Survived'] == 0]['Pclass']], 
        stacked=True, color = ['g','r'],
         bins = 20, label = ['Survived','Dead'])
plt.xlabel('Pclass')
plt.ylabel('Number of passengers')
plt.legend();
plt.show()



#Get to know which class was the cheapest
train.groupby('Pclass').mean()[['Fare']].plot(kind='bar', figsize=(10, 5),  stacked=True)

#Comparing the rate of dead people with the fact of having siblings on board
fig = plt.figure(figsize=(10, 5))
plt.hist([train[train['Survived'] == 1]['SibSp'], train[train['Survived'] == 0]['SibSp']], 
        stacked=True, color = ['g','r'],
         bins = 20, label = ['Survived','Dead'])
plt.xlabel('SibSp')
plt.ylabel('Number of passengers')
plt.legend();
plt.show()

#Comparing the rate of dead people by the fact of having parents on board
plt.hist([train[train['Survived'] == 1]['Parch'], train[train['Survived'] == 0]['Parch']], 
        stacked=True, color = ['g','r'],
         bins = 20, label = ['Survived','Dead'])
plt.xlabel('Parch')
plt.ylabel('Number of passengers')
plt.legend();
plt.show()

#Comparing the rate of dead people by where they embarked
embarked = pd.DataFrame([train[train['Survived'] == 1]['Embarked'].value_counts(), train[train['Survived'] == 0]['Embarked'].value_counts()])
embarked.index = ['Survived','Dead']
embarked.plot(kind='bar',stacked=True, figsize=(10,5))
#Get to know the relation between the ticket price and where they embarked
train.groupby('Embarked').mean()[['Fare']].plot(kind='bar', figsize=(10, 5),  stacked=True)

#Data Cleaning
#Train dataset
df1=train.drop(['Name','Ticket','Cabin','PassengerId','Died'], axis=1)
df1.head()

df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df1.isnull().sum()
df1.dropna(inplace=True)

#Test dataset
#Data cleaning

df2=test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df2.Sex=df2.Sex.map({'female':0, 'male':1})
df2.Embarked=df2.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df2['Age'] = df2['Age'].fillna(df2['Age'].median())

df2.isnull().sum()
df2['Fare']=df2['Fare'].fillna(df2['Fare'].median())
#Data Modelling
y = df1['Survived']
X = df1.drop(['Survived'], axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print("The accuracy for Logistic regression is ", accuracy_score(y_test, y_pred))
pred = logreg.predict(df2)

#K-Nearest Neighbor

KN=KNeighborsClassifier(n_neighbors=3)
KN.fit(X_train,y_train)
pred=KN.predict(X_test)
print("The accuracy for K-Nearest Neighbor Model is ",accuracy_score(pred,y_test))

#SVM

svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
pred=svm.predict(X_test)
print("Accuracy of Support Vector Machine Model is:",accuracy_score(pred,y_test))

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })
submission.to_csv('submission.csv', index=False)