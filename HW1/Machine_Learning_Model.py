import numpy as np; import pandas as pd; import time
import seaborn as sbn; import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
np.set_printoptions(suppress=True)

#=====================================================================
# Upload a Dataset: print(dir(datasets))
# load_iris, load_wine, load_breast_cancer, ...
#=====================================================================
data_read = datasets.load_wine();  #print(data_read.keys())

X = data_read.data
y = data_read.target
dataname = "wine.csv"
targets  = data_read.target_names
features = data_read.feature_names

#---------------------------------------------------------------------
# SETTING
#---------------------------------------------------------------------
N,d = X.shape; nclass=len(set(y));
print('DATA: N, d, nclass =',N,d,nclass)
rtrain = 0.7e0; run = 50; CompEnsm = 2;

def multi_run(clf,X,y,rtrain,run):
    t0 = time.time(); acc = np.zeros([run,1])
    for it in range(run):
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, train_size=rtrain, random_state=it, stratify = y)
        clf.fit(Xtrain, ytrain);
        acc[it] = clf.score(Xtest, ytest)
    etime = time.time()-t0
    return np.mean(acc)*100, np.std(acc)*100, etime # accmean,acc_std,etime

#=====================================================================
# My Classifier
#=====================================================================
from myclf import *    # My Classifier = MyCLF()
if 'MyCLF' in locals():
    accmean, acc_std, etime = multi_run(MyCLF(mode=1),X,y,rtrain,run)

    print('%s: MyCLF()      : Acc.(mean,std) = (%.2f,%.2f)%%; E-time= %.5f'
           %(dataname,accmean,acc_std,etime/run))

#=====================================================================
# Scikit-learn Classifiers, for Comparisions && Ensembling
#=====================================================================
if CompEnsm >= 1:
    exec(open("sklearn_classifiers.py").read())
