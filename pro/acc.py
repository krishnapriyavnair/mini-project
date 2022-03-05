# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:53:48 2022

@author: User
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import metrics,svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import seaborn as sn

data =pd.read_csv("D:\MCA\SEMESTER 3\Mini Project/slope.csv")
print(data.head())
X=data.iloc[:,0:-2]
Y=data.iloc[:,-2]
X.head()

Y.head()

Y.describe()

print("count of ones: ",np.count_nonzero(Y))
print("count of zeros: ",Y.count()-np.count_nonzero(Y) )
Y.plot.hist()


sns.violinplot(y=data["H"])


sns.violinplot(y=data["β"])


sns.violinplot(y=data["γ"])


sns.violinplot(y=data["C"])


sns.violinplot(y=data["Φ"])


ax3 = sns.violinplot(y=data["ru"])

#heatmap 
corrMatrix = X.corr()
heat_map=sn.heatmap(corrMatrix, annot=True)


X_temp=X
# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split 
  
# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0) 
  
# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 

#SVC
from sklearn.svm import SVC
cost= {1, 2, 4, 8, 16, 32}
sigma= {0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.25, 1}

for c in cost:
#     for s in sigma:
    svc=SVC(C = c)
    svc.fit(X_train,y_train)
    pred=svc.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)
    print(score)
    print(c)
    
    svc = SVC(kernel = 'linear')
    svc.fit(X_train, y_train)

    pred=svc.predict(X_test)

    pred=svc.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    metrics.plot_confusion_matrix(svc, X_test,y_test)
    
    ns_probs = [0 for _ in range(len(y_test))]
    # fit a model
    model = SVC(kernel = 'linear',probability=True)
    model.fit(X_train, y_train)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    metrics.plot_confusion_matrix(svc, X_test,y_test)
    # predict probabilities
    lr_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    # print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('SVC: AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='SVC')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred  =  classifier.predict(X_test)
    y_pred
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)
    metrics.plot_roc_curve(rf,X_test,y_test)
    

    ns_probs = [0 for _ in range(len(y_test))]
    # fit a model
    model  = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train, y_train)
    # predict probabilities
    lr_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    # print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Random Forest: AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    from sklearn.ensemble import RandomForestClassifier  
    classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
    classifier.fit(X_train, y_train)  
    y_pred= classifier.predict(X_test)  
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    
    #nb
    X=data.iloc[:,0:-2]
    y=data.iloc[:,-2]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=1)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    y_pred  =  classifier.predict(X_test)
    y_pred
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))