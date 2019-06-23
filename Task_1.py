# Importing libraries
import numpy as np
import pandas as pd
import random
from CustomImputer import CustomImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

'''************************************ Importing Dataset and Split X , Y ********************************************'''
# Importing dataset (train & validation)
train = pd.read_csv('training.csv', sep =";" , thousands=r',' )
validation = pd.read_csv('validation.csv' , sep =";" ,  thousands=r',')
# Split dataset int X_train , Y_train , X_valid and Y_valid
X_train = train.iloc[:, :-1]
Y_train = train.iloc[:, -1]
X_valid = validation.iloc[: , :-1]
Y_valid = validation.iloc[: , -1]


'''******************************** Dealing with Missing Data ************************************************'''
# Dealing with missing numirical data in X_train
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X_train.values[ : , [1,2,7,10,13,14,15,17]])
X_train.iloc[ : , [1,2,7,10,13,14,15,17]] = imputer.transform(X_train.iloc[ : , [1,2,7,10,13,14,15,17]])
# Dealing with missing numirical data in X_valid
imputer = imputer.fit(X_valid.values[ : , [1,2,7,10,13,14,15,17]])
X_valid.iloc[ : , [1,2,7,10,13,14,15,17]] = imputer.transform(X_valid.iloc[ : , [1,2,7,10,13,14,15,17]])
# Dealing with missing catigorical data in X_train , X_valid with custom imputer
Num_cols = len (X_train.columns)
for i in range (Num_cols):
    if(X_train.iloc[ : , i].dtype == np.dtype('O')):
        a  = CustomImputer()   
        a.fit(X_train.iloc[: , i])              
        X_train.iloc[ : , i] = a.transform(X_train.iloc[: ,i])
    if(X_valid.iloc[ : , i].dtype == np.dtype('O')):
        a  = CustomImputer()   
        a.fit(X_valid.iloc[: , i])              
        X_valid.iloc[ : , i] = a.transform(X_valid.iloc[: ,i])
  

'''******************************** Encoding Catigorical Data ************************************************'''
# Deal with Catigorical data in Y_train and Y_valid
labelencoder_Y = LabelEncoder()
Y_train = labelencoder_Y.fit_transform(Y_train)
Y_valid = labelencoder_Y.fit_transform(Y_valid)
# Deal with Catigorical data in X_train and X_valid :
# 1 - add dummy variables to X_train and X_valid
for i in range (Num_cols):
    if(X_train.iloc[ : , i].dtype == np.dtype('O')):
        dummies = pd.get_dummies(X_train.iloc[ : , i])
        X_train = pd.concat([X_train , dummies] , axis = 'columns')
        
    if(X_valid.iloc[ : , i].dtype == np.dtype('O')):
        dummies = pd.get_dummies(X_valid.iloc[ : , i])
        X_valid = pd.concat([X_valid , dummies] , axis = 'columns')
# 2 - add columns of dummy variables exists in X_train and not exists in X_valid in the same index with 0 values        
Ncol = len (X_train.columns)
for i in range (Ncol):
    if (X_train.columns[i] in X_valid.columns) == 0:
        X_valid.insert(i+1, X_train.columns[i], 0)
        
# 3 - remove catigorical columns in X_train and X_valid by using temp variables
X_t = X_train
X_v = X_valid
for i in range (Num_cols):
    if(X_train.iloc[ : , i].dtype == np.dtype('O')):
        X_t = X_t.drop([X_train.columns[i]] , axis = 'columns' )

    if(X_valid.iloc[ : , i].dtype == np.dtype('O')):
        X_v = X_v.drop([X_valid.columns[i]] , axis = 'columns')
X_train = X_t
X_valid = X_v


'''*************************** Feature Scaling for X_train and X_valid ************************************************'''
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.transform(X_valid) 


'''*************************** Train and predict data with diffrent methods ************************************************'''
# Train and predict data with Classification Neural Network
from keras import Sequential
from keras.regularizers import l2
from keras.layers import Dense
random.seed( 3 )
classifier = Sequential()
classifier.add(Dense(5, activation='relu', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05), kernel_initializer= 'random_normal', input_dim=50))
classifier.add(Dense(5, activation='relu', kernel_regularizer=l2(0.05), bias_regularizer=l2(0.05) , kernel_initializer= 'random_normal' ))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer= 'random_normal' ))
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
classifier.fit(X_train,Y_train, batch_size=10, epochs=20)
eval_model=classifier.evaluate(X_train, Y_train)
y_pred=classifier.predict(X_valid)
y_pred =(y_pred>0.5)
cm_NN = confusion_matrix(Y_valid, y_pred)
acc_NN = ((cm_NN[0,0] + cm_NN[1,1])/200) * 100
print("Accuracy with Nueral Network Classifier is " , acc_NN, '%')


# Train and predict data with Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred  = classifier.predict(X_valid)
cm_Log = confusion_matrix(Y_valid , Y_pred)
acc_log = ((cm_Log[0,0] + cm_Log[1,1])/200) * 100
print("Accuracy with logistic classifier is " , acc_log , '%')


# Train and predict data with K-Nearest Neighbors Classifier (KNN)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_valid)
cm_KNN = confusion_matrix(Y_valid, y_pred)
acc_KNN = ((cm_KNN[0,0] + cm_KNN[1,1])/200) * 100
print("Accuracy with K-Nearest Neighbors Classifier (KNN) is " , acc_KNN , '%')


# Train and predict data with Support Vector Classifier (SVC)
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_valid)
cm_SVC = confusion_matrix(Y_valid, y_pred)
acc_SVC = ((cm_SVC[0,0] + cm_SVC[1,1])/200) * 100
print("Accuracy with Support Vector Classifier (SVC) is " , acc_SVC , '%')


# Train and predict data with RandomForest Classifier 
from sklearn.ensemble import RandomForestClassifier
classi = RandomForestClassifier(n_estimators = 40 , criterion = 'entropy' , random_state = 0)
classi.fit(X_train , Y_train) 
y_pred = classi.predict(X_valid)
cm_RF = confusion_matrix(Y_valid, y_pred)
acc_RF = ((cm_RF[0,0] + cm_RF[1,1])/200) * 100
print("Accuracy with RandomForest Classifier is " , acc_RF , '%')






