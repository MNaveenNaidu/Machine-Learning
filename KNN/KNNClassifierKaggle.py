# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:32:36 2019

@author: Naveen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

dataset=pd.read_csv("C:\\Users\\IBM_ADMIN\\Downloads\\Algo\Data\\diabetes.csv")
dataset.iloc[0:6,:]


X=dataset.drop(['Outcome'],axis=1).values
y=dataset['Outcome']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Perform GridSearch to get optimal value of K
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
param_grid={'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()
G_cv=GridSearchCV(knn,param_grid,cv=5)
G_cv.fit(X_train,y_train)
G_cv.best_score_
G_cv.best_params_

 knn=KNeighborsClassifier(n_neighbors=18,metric='minkowski',p=2)
 knn.fit(X_train,y_train)
 y_pred=knn.predict(X_test)
 from sklearn import metrics
 print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
 
#Write a loop to get Optimal number of K with accurancy 
error=[]
for i in range(1,20):
   knn=KNeighborsClassifier(n_neighbors=i)
   knn.fit(X_train,y_train)
   y_pred=knn.predict(X_test)
   from sklearn import metrics
   print("Accuracy:",i,metrics.accuracy_score(y_test, y_pred))
   error.append(np.mean(y_test!=y_pred))
   

plt.plot(range(1,20),error)
plt.xlabel('K')
plt.ylabel("Error")
plt.show()

