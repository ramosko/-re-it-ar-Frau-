import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
import openpyxl




# data = pd.read_csv('creditcard.csv', index_col= None)
# data= openpyxl.load_workbook('creditcard.xlsx')
data=pd.read_csv('creditcard.csv')
# data = pd.read_csv('data.csv', index_col= None)

print(len(data))
print(len(data.loc[data["Class"] == 1]))
print(len(data.loc[data["Class"] == 0]))

testData = data.iloc[: 57000]
trainData = data.iloc[57000: ]


X = trainData.drop(columns = ['Class']).to_numpy()

y = trainData['Class'].to_numpy() 
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_
print(kmeans.labels_)

testY = testData['Class'].to_numpy() 
testX = testData.drop(columns = ["Class"]).to_numpy()

pred = kmeans.predict(testX)
counter = 0
for i in range(len(pred)):
    if(pred[i] == testY[i]):
        counter = counter + 1
print("counter = ", counter / len(testY))


neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X, y)
pred2 = neigh.predict(testX)
counter2 = 0
for i in range(len(pred2)):
    if(pred2[i] == testY[i]):
        counter2 = counter2 + 1
print("counter = ", counter2 / len(testY))


logr = linear_model.LogisticRegression()
logr.fit(X,y)
predicted = logr.predict(testX)
print("Accuracy:",metrics.accuracy_score(testY, predicted))

logr2 = svm.SVC(kernel='poly')
logr2.fit(X,y)
predicted2 = logr.predict(testX)
print("Accuracy:",metrics.accuracy_score(testY, predicted))




