<pre>
"""
Naive Bayes - Gaussian
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
dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-test.csv",header=None)
# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)

arrTrain = dfTrain.as_matrix().astype(int)

y0x1 = arrTrain[:,[1,2]][np.where(arrTrain[:,1]==0)]
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


labelTrain = arrTrain[:,1].astype(int) 
featuresTrain = arrTrain[:,[2,3,4,5]]
gnb = GaussianNB(

).fit(featuresTrain, labelTrain)

arrTest = dfTest.as_matrix().astype(int)
predTest = arrTest[:,1:]
featuresTest = arrTest[:,[2,3,4,5]]

predLabelTest = gnb.predict(featuresTest)
print(predLabelTest)



arrTestLabel0Count = len(arrTest[np.where(arrTest[:,1]==0)])
print('Prob(Label=0) :',arrTestLabel0Count/10000)
print('Prob(Label=1) :',1- arrTestLabel0Count/10000)
actualLabelTest=arrTest[:,1]
arrLabelTest = np.stack([actualLabelTest, predLabelTest]).T
accuracyRate = accuracy_score(actualLabelTest, predLabelTest)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)


</pre>
