#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Topic :  Evaluation for Classification and Regression
Program Code Name : evaluation-for-classification-regression.py
Source : http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
Modified  by Yibeck Lee(yibeck.lee@gmail.com)
"""

print(__doc__)

# ==============================#
# Evaluation for Classification #
# ==============================#
# Accuracy classification score
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

print("[accuracy score] ",accuracy_score(y_true, y_pred, normalize=True))
# [accuracy score]  0.5

print("[accuracy score] ", accuracy_score(y_true, y_pred, normalize=False))
# [accuracy score]  2

# Compute Area Under the Curve (AUC) using the trapezoidal rule
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
print("[Area Under Curve]", metrics.auc(fpr, tpr))
# [Area Under Curve] 0.75

# Compute average precision (AP) from prediction scores
import numpy as np
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print("[Average Precision Score] ",average_precision_score(y_true, y_scores))
# [Average Precision Score]  0.833333333333

# Compute the Brier score
import numpy as np
from sklearn.metrics import brier_score_loss
y_true = np.array([0, 1, 1, 0])
y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
y_prob = np.array([0.1, 0.9, 0.8, 0.3])
print("[brier score loss] ",brier_score_loss(y_true, y_prob))
# [brier score loss]  0.0375

print("[brier score loss] ", brier_score_loss(y_true, 1-y_prob, pos_label=0))
# [brier score loss]  0.0375

print("[brier score loss] ", brier_score_loss(y_true_categorical, y_prob, pos_label="ham"))
# [brier score loss]  0.0375

print("[brier score loss] ", brier_score_loss(y_true, np.array(y_prob) > 0.5))
# [brier score loss]  0.0

# Build a text report showing the main classification metrics
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print("[Classification Metrics]\n",classification_report(y_true, y_pred, target_names=target_names))
#              precision    recall  f1-score   support
#     class 0       0.50      1.00      0.67         1
#     class 1       0.00      0.00      0.00         1
#     class 2       1.00      0.67      0.80         3
# avg / total       0.70      0.60      0.61         5

# Compute confusion matrix to evaluate the accuracy of a classification
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print("[Confusion Matrix]\n",confusion_matrix(y_true, y_pred))
# [Confusion Matrix]
#  [[2 0 0]
#  [0 0 1]
#  [1 0 2]]

# confusion matrix
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
print("[Confusion Matrix]\n",confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"]))
# [Confusion Matrix]
#  [[2 0 0]
#  [0 0 1]
#  [1 0 2]]


evaluation = confusion_matrix(y_true=[0, 1, 0, 1], y_pred=[1, 1, 1, 0])
print("[TN,FP] ",evaluation[0])
# [TN,FP]  [0 2]
print("[FN,TP] ",evaluation[1])
# [FN,TP]  [1 1]

[[tn, fp], [fn, tp]] = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0])
print("[tn, fp]",[tn,fp])
# [tn, fp] [0, 2]
print("[fn, tp]",[fn,tp])
# [fn, tp] [1, 1]


tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
print("[TN(=true negative)]",tn)
# [TN(=true negative)] 0
print("[FP(=false positive)]",fp)
# [FP(=false positive)] 2
print("[FN(=false negative)]",fn)
# [FN(=false negative)] 1
print("[TP(=true positive)]",tp)
# [TP(=true positive)] 1

# Compute the F1 score, also known as balanced F-score or F-measure
from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print("[F1-Macro] ",f1_score(y_true, y_pred, average='macro'))
# [F1-Macro]  0.266666666667
print("[F1-Micro] ",f1_score(y_true, y_pred, average='micro'))
# [F1-Micro] 0.333333333333
print("[F1-Weighted] ",f1_score(y_true, y_pred, average='weighted'))
# [F1-Weighted]  0.266666666667
print("[F1]",f1_score(y_true, y_pred, average=None))
# [F1] [ 0.8  0.   0. ]

# Compute the F-beta score
from sklearn.metrics import fbeta_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print("[F-Beta-Micro] ",fbeta_score(y_true, y_pred, average='micro', beta=0.5))
# [F-Beta-Micro]  0.333333333333
print("[F-Beta-Weighted] ",fbeta_score(y_true, y_pred, average='weighted', beta=0.5))
# [F-Beta-Weighted]  0.238095238095
print("[F-Beta] ",fbeta_score(y_true, y_pred, average=None, beta=0.5))
# [F-Beta]  [ 0.71428571  0.          0.        ]

# Compute the average Hamming loss
from sklearn.metrics import hamming_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
print("[Hamming Loss] ",hamming_loss(y_true, y_pred))
# [Hamming Loss]  0.25

print("[Hamming Loss] ",hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2))))
# [Hamming Loss]  0.75

# Average hinge loss (non-regularized)
from sklearn import svm
from sklearn.metrics import hinge_loss
X = [[0], [1]]
y = [-1, 1]
est = svm.LinearSVC(random_state=0)
est.fit(X, y)
# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#      verbose=0)
pred_decision = est.decision_function([[-2], [3], [0.5]])
print("[Prediction Decision]",pred_decision)  
# [Prediction Decision] [-2.18177262  2.36361684  0.09092211]
print("[Hinge Loss]",hinge_loss([-1, 1, 1], pred_decision))
# [Hinge Loss] 0.303025963688

# hinge_loss in the multiclass case
X = np.array([[0], [1], [2], [3]])
Y = np.array([0, 1, 2, 3])
labels = np.array([0, 1, 2, 3])
est = svm.LinearSVC()
est.fit(X, Y)
# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)
pred_decision = est.decision_function([[-1], [2], [3]])
y_true = [0, 2, 3]
print("[Hinge Loss]",hinge_loss(y_true, pred_decision, labels))
# [Hinge Loss] 0.564104824127

# Jaccard similarity coefficient score
import numpy as np
from sklearn.metrics import jaccard_similarity_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print("[Jaccard Similarity] ",jaccard_similarity_score(y_true, y_pred,normalize=True))
# [Jaccard Similarity]  0.5

print("[Jaccard Similarity] ",jaccard_similarity_score(y_true, y_pred, normalize=False))
# [Jaccard Similarity]  2

print("[Jaccard Similarity] ",jaccard_similarity_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))
# [Jaccard Similarity]  0.75

# Log loss(= logistic loss = cross-entropy loss)
from sklearn.metrics import log_loss
print("[Log Loss]",log_loss(["spam", "ham", "ham", "spam"],  [[.1, .9], [.9, .1], [.8, .2], [.35, .65]]))
# [Log Loss] 0.216161874681

# Compute the Matthews correlation coefficient (MCC)
from sklearn.metrics import matthews_corrcoef
y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]
print("[Matthews Correcoef]",matthews_corrcoef(y_true, y_pred))
# [Matthews Correcoef] -0.333333333333

# precision recall curve
# Compute precision-recall pairs for different probability thresholds
import numpy as np
from sklearn.metrics import precision_recall_curve
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print("[Precision]",precision)  
# [Precision] [ 0.66666667  0.5         1.          1.        ]
print("[Recall]",recall)
# [Recall] [ 1.   0.5  0.5  0. ]
print("[Thresholds] ",thresholds)
# [Thresholds]  [ 0.35  0.4   0.8 ]

# precision recall fscore support
from sklearn.metrics import precision_recall_fscore_support
y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
print("[Precision Recall Fscore Support-Macro]",precision_recall_fscore_support(y_true, y_pred, average='macro'))
# [Precision Recall Fscore Support-Macro] (0.22222222222222221, 0.33333333333333331, 0.26666666666666666, None)
print("[Precision Recall Fscore Support-Micro]",precision_recall_fscore_support(y_true, y_pred, average='micro'))
# [Precision Recall Fscore Support-Micro] (0.33333333333333331, 0.33333333333333331, 0.33333333333333331, None)
print("[Precision Recall Fscore Support-Weighted]",precision_recall_fscore_support(y_true, y_pred, average='weighted'))
# [Precision Recall Fscore Support-Weighted] (0.22222222222222221, 0.33333333333333331, 0.26666666666666666, None)

# precision score
from sklearn.metrics import precision_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print("[Precision-Macro]",precision_score(y_true, y_pred, average='macro'))
# [Precision-Macro] 0.222222222222

print("[Precision-Micro]",precision_score(y_true, y_pred, average='micro'))
# [Precision-Micro] 0.333333333333

print("[Precision-Weighted]",precision_score(y_true, y_pred, average='weighted'))
# [Precision-Weighted] 0.222222222222

print("[Precision]",precision_score(y_true, y_pred, average=None))
# [Precision] [ 0.66666667  0.          0.        ]

# recall score
from sklearn.metrics import recall_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print("[Recall-Macro] ",recall_score(y_true, y_pred, average='macro'))
# [Recall-Macro]  0.333333333333
print("[Recall-Micro] ",recall_score(y_true, y_pred, average='micro'))
# [Recall-Micro]  0.333333333333
print("[Recall-Weighted] ",recall_score(y_true, y_pred, average='weighted'))
# [Recall-Weighted]  0.333333333333
print("[Recall] ",recall_score(y_true, y_pred, average=None))
# [Recall]  [ 1.  0.  0.]


# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction score
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print("[ROC-AUC]",roc_auc_score(y_true, y_scores))
# [ROC-AUC] 0.75

# Compute Receiver operating characteristic (ROC)
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print("[FPR]",fpr)
# [FPR] [ 0.   0.5  0.5  1. ]

print("[TPR]",tpr)
# [TPR] [ 0.5  0.5  1.   1. ]

print("[Thresholds]",thresholds)
# [Thresholds] [ 0.8   0.4   0.35  0.1 ]

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.show()




# Zero-one classification loss
from sklearn.metrics import zero_one_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]
print("[Zero One Loss]",zero_one_loss(y_true, y_pred, normalize=True))
# [Zero One Loss] 0.25
print("[Zero One Loss]",zero_one_loss(y_true, y_pred, normalize=False))
# [Zero One Loss] 1
print("[Zero One Loss]",zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))
# [Zero One Loss] 0.5



# Explained variance regression score function
from sklearn.metrics import explained_variance_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("[Explaned Variance]",explained_variance_score(y_true, y_pred))
# [Explaned Variance] 0.957173447537

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print("[Explaned Variance]",explained_variance_score(y_true, y_pred, multioutput='uniform_average'))
# [Explaned Variance] 0.983870967742


# ==============================#
# Evaluation for Regression     #
# ==============================#
# MSE(Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)
# 0.5
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print("[MAE]",mean_absolute_error(y_true, y_pred))
# [MAE] 0.75
print("[MAE]",mean_absolute_error(y_true, y_pred, multioutput='raw_values'))
# [MAE] [ 0.5  1. ]
print("[MAE]",mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7]))
# [MAE] 0.85


# MSE(Mean Squared Error)
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)
# 0.375
y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]
print("[MSE]",mean_squared_error(y_true, y_pred))  
# [MSE] 0.708333333333

print("[MSE]",mean_squared_error(y_true, y_pred, multioutput='raw_values'))
# [MSE] [ 0.41666667  1.        ]
print("[MSE]",mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7]))
# [MSE] 0.825

# Mean Squared Log Error
from sklearn.metrics import mean_squared_log_error
y_true = [3, 5, 2.5, 7]
y_pred = [2.5, 5, 4, 8]
print("[MSLE]",mean_squared_log_error(y_true, y_pred))  
# [MSLE] 0.0397301229846

y_true = [[0.5, 1], [1, 2], [7, 6]]
y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
print("[MSLE]",mean_squared_log_error(y_true, y_pred))
# [MSLE] 0.0441993618892
print("[MSLE]",mean_squared_log_error(y_true, y_pred, multioutput='raw_values'))
# [MSLE] [ 0.00462428  0.08377444]
print("[MSLE]",mean_squared_log_error(y_true, y_pred, multioutput=[0.3, 0.7]))
# [MSLE] 0.0600293941797

# Median Absolute Error
from sklearn.metrics import median_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("[Median Absolute Error]",median_absolute_error(y_true, y_pred))
# [Median Absolute Error] 0.5

# R-Square (coefficient of determination) regression score function.
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("[R-Square]",r2_score(y_true, y_pred))
# [R-Square] 0.948608137045

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print("[R-Square]",r2_score(y_true, y_pred, multioutput='variance_weighted'))
# [R-Square] 0.938256658596

y_true = [1,2,3]
y_pred = [1,2,3]
print("[R-Square]",r2_score(y_true, y_pred))
# [R-Square] 1.0
y_true = [1,2,3]
y_pred = [2,2,2]
print("[R-Square]",r2_score(y_true, y_pred))
# [R-Square] 0.0
y_true = [1,2,3]
y_pred = [3,2,1]
print("[R-Square]",r2_score(y_true, y_pred))
# [R-Square] -3.0


