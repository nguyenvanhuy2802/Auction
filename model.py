# Neural Network, Decision Trees, Linear Regression, Random Forest, SVM.
import numpy as np
import pandas as pd
from sklearn import *
from abc import ABC,abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

class Model(ABC):
  def __init__(self,data,targetData,model):
    self.model = model
    self.target = targetData
    self.dataOfSet = data
    [self.x_train,self.x_test,self.y_train,self.y_test] = train_test_split(self.dataOfSet,self.target,random_state=0,test_size=0.3)
    self.trainModel(self.x_train,self.y_train)
    super().__init__()

  def trainModel(self,x_train,y_train):
    self.model.fit(x_train,y_train)

  def getYPredict(self):
      return  self.model.predict(self.x_test)

  def getYPredictWithout(self,x_test_data):
      return self.model.predict(x_test_data)

  def getAccurancy(self):
    y_predict = self.getYPredict()
    accuracyScore=  round(accuracy_score(self.y_test,y_predict),4)
    precisionScore= round(precision_score(self.y_test,y_predict,average="micro"),4)
    recall = round(recall_score(self.y_test,y_predict,average="micro"),4)
    f1= round(f1_score(self.y_test,y_predict,average= "micro"),4)
    return [accuracyScore,precisionScore,recall,f1]


class SVMModel(Model):
  def __init__(self, data,targetData,kernelData):
    self.svm_model = svm.SVC(kernel=kernelData)
    super().__init__(data,targetData, self.svm_model)

 
class DecisionTreeModel(Model):
  def __init__(self, data,targetData):
    self.decision_model = tree.DecisionTreeClassifier(criterion="gini",  max_depth=3, min_samples_leaf=5)
    super().__init__(data,targetData, self.decision_model)
      
class RandomModel(Model):
  def __init__(self,data,tartgetData,n_estimators,max_features,max_depth,max_leaf_nodes):
    self.randomforest_model = ensemble.RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes)
    super().__init__(data,tartgetData,self.randomforest_model) 
  
class LogisticModel(Model):
  def __init__(self,data,tartgetData):
    self.logistic_model = linear_model.LogisticRegression()
    super().__init__(data,tartgetData,self.logistic_model) 


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def CNN_MODEL(X_scaled,label):
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, label, test_size=0.3, random_state=0)

# Xây dựng mô hình CNN
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)))
  model.add(MaxPooling2D(pool_size=(2, 1)))
  model.add(Conv2D(64, kernel_size=(3, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 1)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Hiển thị kiến trúc của mô hình
  model.summary()
  model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
  y_pred_prob = model.predict(X_test)  # Dự đoán xác suất
  y_pred = (y_pred_prob > 0.5).astype(int) 
   
  accuracy_cnn = round(accuracy_score(y_test, y_pred),4)
  precision_cnn = round(precision_score(y_test, y_pred),4)
  recall_cnn = round(recall_score(y_test, y_pred),4)
  f1_cnn = round(f1_score(y_test, y_pred) ,4)


  return [accuracy_cnn,precision_cnn,recall_cnn,f1_cnn]
  