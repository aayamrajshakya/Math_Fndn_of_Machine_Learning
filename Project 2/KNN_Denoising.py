# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#---------------------------------------------------------------
import numpy as np; np.set_printoptions(suppress=True)
from sklearn.neighbors import KNeighborsClassifier

#----------------------------------
def conf_count(neigh,y):
    cscore = np.zeros(y.shape)
    for i in range(len(y)):
        labels = y[neigh[i]]
        cscore[i] = len(labels[labels==y[i]])
    return cscore

#-- Data Generation ---------------
n=5; d=3;
X0 = np.random.random((n,d)); X1 = 2*np.random.random((n,d))+0.2;
X  = np.row_stack((X0,X1))
y0 = np.zeros(n,); y1 = np.ones(n,);
y  = np.row_stack((y0,y1)).reshape((len(X)))
print('(X,y):\n',np.column_stack((X,y)))

#-- Parameters for KNN ------------
k=5; xi=4;

#-- KNN, with "kneighbors" --------
KNN = KNeighborsClassifier(n_neighbors=k);
KNN.fit(X,y)

neigh = KNN.kneighbors(X, return_distance=False)
print('Kneighbors:\n',neigh)

#-- Confidence score --------------
cscore = conf_count(neigh,y)
print('Confidence Score:\n',cscore)
xi -= 0.1;
Xd = X[cscore>=xi]
yd = y[cscore>=xi]
print('Denoised Data (Xd,yd):\n',np.column_stack((Xd,yd)))

