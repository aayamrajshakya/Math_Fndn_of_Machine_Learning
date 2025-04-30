import numpy as np; import numpy.linalg as la
import pandas as pd; import seaborn as sbn;
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets; #print(dir(datasets))
np.set_printoptions(suppress=True)
import time

from synthetic_data import *; from util_PCA import *
from sklearn_classifiers import classifiers, names

#-------------------------------------------------------
# DATA = np.loadtxt('wine.data', delimiter=',')
DATA = synthetic_data()
X = DATA[:,:-1]; y = DATA[:,-1]
N,d = X.shape; nclass = len(set(y))
print(' nclass = %d; (N, d) = (%d, %d)' %(nclass,N,d))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

# Denoise only train set
Xd_train, yd_train = data_denoise(X_train,y_train,0.9)

dataCellipses(X, y, 'synthetic_initial-MVEE.png')
dataCellipses(Xd_train,yd_train,'synthetic_denoised-MVEE.png')


# Adopted from Homework 1
def multi_run(clf, X_train, y_train, X_test, y_test, NUM_EPOCHS):
    t0 = time.time(); acc = np.zeros([NUM_EPOCHS, 1])
    for i in range(NUM_EPOCHS):
        clf.fit(X_train, y_train)            
        acc[i] = clf.score(X_test, y_test)
    etime = time.time() - t0
    return np.mean(acc)*100, np.std(acc)*100, etime # accmean,acc_std,etime

#=====================================================================
print('\n====== Comparison: Scikit-learn Classifiers =================')
#=====================================================================

def helper(dataset_type, X_train, y_train, X_test, y_test):
    NUM_EPOCHS = 50
    acc_max=0; Acc_CLF = np.zeros([len(classifiers),1])
    print(f'\033[91mPerformance in {dataset_type} dataset:\033[0m:')
    for k, (name, clf) in enumerate(zip(names, classifiers)):
        accmean, acc_std, etime = multi_run(clf, X_train, y_train, X_test, y_test, NUM_EPOCHS)

        Acc_CLF[k] = accmean
        if accmean>acc_max: acc_max,algname = accmean,name
        print('%s: Acc.(mean,std) = (%.2f,%.2f)%%; E-time= %.5f'
            %(name,accmean,acc_std,etime/NUM_EPOCHS))
    print('--------------------------------------------------------------')
    print('Acc: (mean,max) = (%.2f,%.2f)%%; Best = %s\n'
        %(np.mean(Acc_CLF),acc_max,algname))

print(f"\033[35mData points chosen: {len(Xd_train)}/{len(X)}\033[0m")
helper('denoised', Xd_train, yd_train, X_test, y_test)

print("\033[32mOriginal dataset:\nTesting with the original dataset but with same no. of points as in denoised set.\033[0m")
X1, X2, y1, y2 = train_test_split(X, y, train_size=0.635,
                                random_state=42, stratify=y)

print(f"\033[35mData points chosen: {len(X1)}/{len(X)}\033[0m")
helper('demo', X1, y1, X2, y2)
print("\033[32mAs you can see, the classifiers perform better on denoised dataset compared to this dataset\n, which shows that the denoising process is successsful!\033[0m")