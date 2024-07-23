from model import SVMModel,DecisionTreeModel,RandomModel,LogisticModel,CNN_MODEL
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np 
from numpy import isnan
import matplotlib as plt
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

data = pd.read_pickle('train_data_processed.pickle')
feature=data.iloc[:,:-1]
label = data.iloc[:,-1]



#  scale data 
scaler = StandardScaler()
scaled_feature = scaler.fit_transform(feature)

table = PrettyTable(['','Accuracy','Precision','Recall','F1'])

svm_kernel =['linear','poly','rbf','sigmoid']
for kernel in svm_kernel :
  svm = SVMModel(scaled_feature,label,kernel)
  accurancy = svm.getAccurancy()
  table.add_row([f'SVM_{kernel}',accurancy[0],accurancy[1],accurancy[2],accurancy[3]])

# 
decisiontree = DecisionTreeModel(scaled_feature,label)
accurancy_decisiontree = decisiontree.getAccurancy()
table.add_row([f'DecisionTree',accurancy_decisiontree[0],accurancy_decisiontree[1],accurancy_decisiontree[2],accurancy_decisiontree[3]])


# Định nghĩa tham số
# param_grid_RDM = {
#     'n_estimators': [25, 50, 100, 150],
#     'max_features': ['sqrt', 'log2', None],
#     'max_depth': [3, 6, 9],
#     'max_leaf_nodes': [3, 6, 9],
# }

random = RandomModel(scaled_feature,label,n_estimators=100,max_features='sqrt',max_depth=9,max_leaf_nodes=9)
accurancy_random = random.getAccurancy()
table.add_row([f'Random Forest',accurancy_random[0],accurancy_random[1],accurancy_random[2],accurancy_random[3]])


# 
logistic = LogisticModel(feature,label)
accurancy_logistic = logistic.getAccurancy()
table.add_row([f'Logistic Regression',accurancy_logistic[0],accurancy_logistic[1],accurancy_logistic[2],accurancy_logistic[3]])



# 
X_scaled = scaled_feature.reshape(scaled_feature.shape[0], scaled_feature.shape[1], 1, 1)
[accuracy_cnn,precision_cnn,recall_cnn,f1_cnn] = CNN_MODEL(X_scaled,label)
table.add_row([f'CNN',accuracy_cnn,precision_cnn,recall_cnn,f1_cnn])

print(table)