<pre>
"""
GLM - ElasticNet
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
from pandas import Series
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-test.csv",header=None)
# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
dfTrain.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
dfTest.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']

print('dfTrain.columns : ',  dfTrain.columns)
print('dfTrain.columns.values : ',  dfTrain.columns.values)

print('[alpha=0]')
glm_elasticnet = lm.ElasticNet(
	  alpha=0
	, l1_ratio=1
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTrain = glm_elasticnet.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y1'],predLabelTrain)
print('r-squared = ', r2)
mse = mean_squared_error(dfTrain['Y1'],predLabelTrain)
print('mse = {:12,.0f}'.format(mse))
elasticnet_coef =  Series(
	 glm_elasticnet.coef_
	,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
	).sort_values()
print('type of elasticnet_coef:', type(elasticnet_coef))
# print(elasticnet_coef.size)
print('beta coefficient of variables')
for i in range(elasticnet_coef.size):
	print('{:4s} : {:10.8f}'.format(elasticnet_coef.index.values[i],elasticnet_coef.values[i]))
plt.figure(1)
elasticnet_coef.plot(kind='bar',grid=True)
plt.savefig("elasticnet_alpha0.png")
print('[alpha=100]')
glm_elasticnet = lm.ElasticNet(
  	  alpha=10
	, l1_ratio=0.5
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTrain = glm_elasticnet.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y1'],predLabelTrain)
print('r-squared = ', r2)
mse = mean_squared_error(dfTrain['Y1'],predLabelTrain)
print('mse = {:12,.0f}'.format(mse))
elasticnet_coef =  Series(
	 glm_elasticnet.coef_
	,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
	).sort_values()
print('type of elasticnet_coef:', type(elasticnet_coef))
# print(elasticnet_coef.size)
print('beta coefficient of variables')
for i in range(elasticnet_coef.size):
	print('{:4s} : {:10.8f}'.format(elasticnet_coef.index.values[i],elasticnet_coef.values[i]))
plt.figure(2)
elasticnet_coef.plot(kind='bar',grid=True)
plt.savefig("elasticnet_alpha100.png")

plt.show()

</pre>
