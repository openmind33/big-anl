#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Title : Descriptive Statistics
Program Code Name : desc-stats.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

import graphviz 
from sklearn import tree

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output

dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv"
    ,header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv"
    ,header=None)
nObsDfTrain = len(dfTrain)
nObsDfTest = len(dfTest)
print(nObsDfTrain,nObsDfTest)

print(dfTrain[0:10]) # 10개의 관찰치 출력
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
# array index :     0    1    2    3   4     5    6    7    8    9  10     11    12
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
dfTest.columns  = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
print(dfTrain.columns.values)
print(dfTrain[['Y1','Y2']][:10])

plt.figure(1)
for col in dfTrain.columns:
    (kurtosis,skew)=dfTrain[col].kurtosis(),dfTrain[col].skew()
    print("{}: 첨도={}, 왜도={}".format(col,kurtosis,skew))
    # dfTrain[[col]].plot.hist(stacked=True)
plt.show()


plt.figure(2)
for col in ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']:
	dfTrain.loc[dfTrain['Y2']==0, col].plot.hist(label="0")
	dfTrain.loc[dfTrain['Y2']==1, col].plot.hist(label="1")
	plt.legend()
	plt.show()


plt.figure(3)
for col in ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']:
	dfTrain.boxplot(column=[col],by='Y2')
	# plt.legend()
	plt.show()


	
"""
"""