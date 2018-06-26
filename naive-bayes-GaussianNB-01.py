#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Naive Bayes - Gaussian
Program Code Name : naive-bayes-GaussianNB-01.py 
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output


dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
nObsDfTrain = len(dfTrain)
print(nObsDfTrain)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)
nObsDfTest = len(dfTest)
print(nObsDfTest)
print(dfTrain[0:10]) # 10개의 관찰치 출력
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])

# array index :     0    1    2    3   4     5    6    7    8    9  10     11    12
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
dfTest.columns  = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
print(dfTrain.columns.values)
arrTrain = dfTrain.as_matrix().astype(int)
"""
y0x1 = arrTrain[:,[1,2]][np.where(arrTrain[:,1]==0)]
print(y0x1)

y1x1 = arrTrain[:,[1,2]][np.where(arrTrain[:,1]==1)]
fig = plt.figure('x1')
n, bins, patches = plt.hist(y0x1, facecolor='g')
n, bins, patches = plt.hist(y1x1, facecolor='r')
plt.show()
y0x2 = arrTrain[:,[1,3]][np.where(arrTrain[:,1]==0)]
y1x2 = arrTrain[:,[1,3]][np.where(arrTrain[:,1]==1)]
fig = plt.figure('x2')
n, bins, patches = plt.hist(y0x2, facecolor='g')
n, bins, patches = plt.hist(y1x2, facecolor='r')

y0x3 = arrTrain[:,[1,4]][np.where(arrTrain[:,1]==0)]
y1x3 = arrTrain[:,[1,4]][np.where(arrTrain[:,1]==1)]
fig = plt.figure('x3')
n, bins, patches = plt.hist(y0x3, facecolor='g')
n, bins, patches = plt.hist(y1x3, facecolor='r')

y0x4 = arrTrain[:,[1,5]][np.where(arrTrain[:,1]==0)]
y1x4 = arrTrain[:,[1,5]][np.where(arrTrain[:,1]==1)]
fig = plt.figure('x4')
n, bins, patches = plt.hist(y0x4, facecolor='g')
n, bins, patches = plt.hist(y1x4, facecolor='r')

y0x5 = arrTrain[:,[1,6]][np.where(arrTrain[:,1]==0)]
y1x5 = arrTrain[:,[1,6]][np.where(arrTrain[:,1]==1)]
fig = plt.figure('x6')
n, bins, patches = plt.hist(y0x5, facecolor='g')
n, bins, patches = plt.hist(y1x5, facecolor='r')

y0x6 = arrTrain[:,[1,7]][np.where(arrTrain[:,1]==0)]
y1x6 = arrTrain[:,[1,7]][np.where(arrTrain[:,1]==1)]
fig = plt.figure('x6')
n, bins, patches = plt.hist(y0x6, facecolor='g')
n, bins, patches = plt.hist(y1x6, facecolor='r')

y0x7 = arrTrain[:,[1,8]][np.where(arrTrain[:,1]==0)]
y1x7 = arrTrain[:,[1,8]][np.where(arrTrain[:,1]==1)]
fig = plt.figure('x7')
n, bins, patches = plt.hist(y0x7, facecolor='g')
n, bins, patches = plt.hist(y1x7, facecolor='r')


plt.show()
"""

arrTrainLabels = arrTrain[:,1:2].astype(int) 
print(arrTrainLabels)
arrTrainFeatures = arrTrain[:,[2,3,4,5]]
print(arrTrainFeatures)
model = GaussianNB().fit(arrTrainFeatures, arrTrainLabels)
print(model)
print(model.get_params())

arrTest = dfTest.as_matrix().astype(int)
arrTestLabels = arrTest[0:1000,1]
arrTestFeatures = arrTest[0:1000,[2,3,4,5]]

pred_proba = model.predict_proba(arrTestFeatures)
print(pred_proba)

pred = model.predict(arrTestFeatures)
print(pred)

for i in range(len(arrTestLabels)):
	print(arrTestLabels[i:i+1,][0],"-", pred[i,])

pred_score = model.score(arrTestFeatures, arrTestLabels)
print(pred_score)

arrTestLabel0Count = len(arrTest[np.where(arrTest[:,1]==0)])
print('Prob(Label=0) :',arrTestLabel0Count/10000)
print('Prob(Label=1) :',1- arrTestLabel0Count/10000)
# actualTestLabels=arrTest[:,1]
# arrTestActualPred = np.stack([arrTestLabels, pred]).T
accuracyRate = accuracy_score(arrTestLabels, pred)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)
"""
"""
