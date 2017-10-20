<pre>
"""
SVM - Classification
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-test.csv",header=None)
# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
print(dfTrain[0:10])
arrTrain = dfTrain.as_matrix().astype(int)
print(arrTrain[0:10])
print(arrTrain.shape)

# print(name_dtype)
arrTrain = np.array(arrTrain)
print(arrTrain.dtype)



print(np.unique(arrTrain[:,1])) # binary label 
numClass = np.unique(arrTrain[:,1]).size
print('numClass',numClass)
labelTrain = arrTrain[:,1].astype(int) 
print(labelTrain[0:100])
print('[labelTrain[0,1] : ', np.bincount(labelTrain)) # get frequency of [0,1]
featuresTrain = arrTrain[:,[2,3]]


# svc = LinearSVC(random_state=0)
# svc.fit(featuresTrain, labelTrain)
# print(svc.coef_)
# print(svc.intercept_)

svc = SVC(
    C=100.0
    # , cache_size=200
    # , class_weight=None
    # , coef0=0.0
    # , decision_function_shape='ovr'
    # , degree=3
    # , gamma='auto'
    , gamma=1000
    , kernel='rbf'
    # , max_iter=-1
    # , probability=False
    # , random_state=None
    # , shrinking=True
    # , tol=0.001
    # , tol=0.001
    # , verbose=False
).fit(featuresTrain, labelTrain)
arrTest = dfTest.as_matrix().astype(int)
# predTest = arrTest[:,1:]
featuresTest = arrTest[:,[2,3]]
predLabelTest = svc.predict(featuresTest)
print(predLabelTest)
actualLabelTest=arrTest[:,1]
print(actualLabelTest)
arrLabelTest = np.stack([actualLabelTest, predLabelTest]).T
print(arrLabelTest)
accuracyRate = accuracy_score(actualLabelTest, predLabelTest)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)
</pre>
