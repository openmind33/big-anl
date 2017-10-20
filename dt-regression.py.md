<pre>
"""
Decision Tree Regression
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-test.csv",header=None)
# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
print(dfTrain[0:10])
arrTrain = dfTrain.as_matrix().astype(int)
print(arrTrain[0:10])
print(arrTrain.shape)
name_dtype = [
      ('continuousLabel', int)
    , ('binaryLabel', int)
    , ('X0', int)
    , ('X1', int)
    , ('X2', int)
    , ('X3', int)
    , ('X4', int)
    , ('X5', int)
    , ('X6', int)
    , ('X7', int)
    , ('X8', int)
    , ('X9', int)
    , ('X10', int)
    ]
# print(name_dtype)
arrTrain = np.array(arrTrain, dtype = name_dtype)
print(arrTrain.dtype)



labelTrain = arrTrain[:,0].astype(int) 
print(labelTrain[0:100])
featuresTrain = arrTrain[:,2:]
# Train and make model
# dtr = DecisionTreeRegressor().fit(featuresTrain, labelTrain)
dtr = DecisionTreeRegressor(
      criterion='mse'
    , max_depth=None
    , max_features=None
    , max_leaf_nodes=None
    , min_impurity_split=0.01
    , min_samples_leaf=1
    , min_samples_split=2
    , min_weight_fraction_leaf=0.0
    , presort=False
    , random_state=None
    , splitter='best'
    ).fit(featuresTrain, labelTrain)
print(dtr)
importances = dtr.feature_importances_
print('feature importances : ', importances)
indices = np.argsort(importances)[::-1]
print('indices : ', indices)

for f in range(featuresTrain.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(featuresTrain.shape[1]), importances[indices],
    color="b", 
    align="center")
plt.xticks(range(featuresTrain.shape[1]), indices)
plt.xlim([-1, featuresTrain.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
plt.show()

arrTest = dfTest.as_matrix().astype(int)

featuresTest = arrTest[:,2:]
predLabelTest = dtr.predict(featuresTest)
#print(predLabelTest)
actualLabelTest=arrTest[:,0]
arrLabelTest = np.stack([actualLabelTest, predLabelTest]).T
print(arrLabelTest[0:10,0])
print(arrLabelTest[0:10,1])

plt.scatter(arrLabelTest[0:10000,0], arrLabelTest[0:10000,1])
plt.show()


"""
import graphviz 
dot_data = tree.export_graphviz(dtc, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dtc") 
dot_data = tree.export_graphviz(dtc, out_file=None, 
                         feature_names=['X0','X1','X2','X3','X4','X5','X6','X7','X8','X0','X10'],  
                         # class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
"""
</pre>
