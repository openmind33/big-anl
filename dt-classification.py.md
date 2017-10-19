<pre>
"""
Decision Tree Classification
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score \
, confusion_matrix,classification_report
from sklearn import tree # tree graph

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output

dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv"
    ,header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-test.csv"
    ,header=None)
# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
print(dfTrain[0:10]) # 10개의 관찰치 출력
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
# array index :     0    1    2    3   4     5    6    7    8    9  10     11    12
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6'
,'V7','V8','V9','V10','V11']
dfTest.columns  = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
print(dfTrain.columns.values)
dtc = DecisionTreeClassifier(
    max_depth=2
    ).fit(dfTrain[['V1','V2','V3','V4','V5'
                    ,'V6','V7','V8','V9','V10','V11']]
         ,dfTrain['Y2'])

print(dtc)
import graphviz 
dot_data = tree.export_graphviz(dtc, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dtc") 
graph
"""
dot_data = tree.export_graphviz(dtc, out_file="tree.dot", 
                         feature_names=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11'],
                         class_names=['good','bad']
                         )
graph = graphviz.Source(dot_data)
graph.render("tree.png") 
graph
# dot -Tpng tree.dot -o tree.png
"""
pred = dtc.predict(dfTest[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']])
print(pred)
print(dfTest['Y2'].count())
dfFreq = pd.value_counts(dfTest['Y2']).to_frame().reset_index()
print(dfFreq)

# print(dfTest['Y2'].describe())

cmatrix = confusion_matrix(y_true=dfTest['Y2'],y_pred=pred)

print(cmatrix)

true_negative, false_positive, false_negative, true_positive \
= confusion_matrix(y_true=dfTest['Y2'],y_pred=pred).ravel()

print('true_negative, false_positive, false_negative, true_positive\n', true_negative, false_positive, false_negative, true_positive)

print(classification_report(y_true=dfTest['Y2'], y_pred=pred))

accuracyRate = accuracy_score(dfTest['Y2'], pred,)

print('정확도 = ',accuracyRate)

print('오분류율 = ', 1 - accuracyRate)


"""
print(dtc)






names = dfTrain[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']].columns.values
print(names)
importances = dtc.feature_importances_
sorted_importances = np.argsort(importances)
print(sorted_importances)

print("변수 중요도")
print(sorted(zip(map(lambda x: round(x, 4), dtc.feature_importances_), names), reverse=True))
padding = np.arange(len(names)) + 0.5
plt.barh(padding, importances[sorted_importances], align='center')
# plt.barh(padding, importances[sorted_importances])
plt.yticks(padding, names[sorted_importances])
plt.xlabel("Relative Importance")
plt.title("Feature Importance")
plt.show()
"""



"""
print('feature importances : ', importances)
indices = np.argsort(importances)[::-1]

print('indices : ', indices)
arrTrain = dfTrain.as_matrix().astype(int)
featuresTrain = arrTrain[:,[0,1,2,3,4,5,6,7,8,9,10]]

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
"""
"""
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



print(np.unique(arrTrain[:,1])) # binary label 
numClass = np.unique(arrTrain[:,1]).size
print('numClass',numClass)
labelTrain = arrTrain[:,1].astype(int) 
print(labelTrain[0:100])
print('[labelTrain[0,1] : ', np.bincount(labelTrain)) # get frequency of [0,1]




featuresTrain = arrTrain[:,2:]
# print('featuresTrain.dtype : ', featuresTrain.dtype[2:10])
print(featuresTrain.shape)
# Train and make model
dtc = DecisionTreeClassifier().fit(featuresTrain, labelTrain)
print(dtc)


arrTest = dfTest.as_matrix().astype(int)
predTest = arrTest[:,1:]
featuresTest = arrTest[:,2:]
predLabelTest = dtc.predict(featuresTest)
print(predLabelTest)
actualLabelTest=arrTest[:,1]
print(actualLabelTest)
arrLabelTest = np.stack([actualLabelTest, predLabelTest]).T
print(arrLabelTest)
accuracyRate = accuracy_score(actualLabelTest, predLabelTest)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)


importances = dtc.feature_importances_
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


import graphviz 
dot_data = tree.export_graphviz(dtc, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dtc") 
dot_data = tree.export_graphviz(dtc, out_file=None, 
                         feature_names=['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10'])  
graph = graphviz.Source(dot_data)  
graph
# print(dtc.predict(arrTrain[:1, :]))
# print(dtc.predict_proba(arrTrain[:1, :]))

"""
"""
# Parameters
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

# Load data
iris = load_iris()

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()

"""
</pre>
