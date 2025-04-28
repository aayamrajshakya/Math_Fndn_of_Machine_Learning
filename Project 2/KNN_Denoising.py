# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#---------------------------------------------------------------
import numpy as np; np.set_printoptions(suppress=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import time
from sklearn_classifiers import classifiers, names
from sklearn.metrics import accuracy_score

#----------------------------------
def conf_count(neigh,y):
    cscore = np.zeros(y.shape)
    for i in range(len(y)):
        labels = y[neigh[i]]
        cscore[i] = len(labels[labels==y[i]])
    return cscore

#-- Data Generation ---------------
# n=5; d=3
# X0 = np.random.random((n,d)); X1 = 2*np.random.random((n,d))+0.2
# X  = np.row_stack((X0,X1))
# y0 = np.zeros(n,); y1 = np.ones(n,)
# y  = np.row_stack((y0,y1)).reshape((len(X)))
# print('(X,y):\n',np.column_stack((X,y)))

data_read = datasets.load_wine()
X = data_read.data
y = data_read.target
dataname = "wine.csv"
targets = data_read.target_names
features = data_read.feature_names

# Standardizing the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                random_state=42, stratify=y)

#-- Parameters for KNN ------------
k=5; xi=4

#-- KNN, with "kneighbors" --------
KNN = KNeighborsClassifier(n_neighbors=k)
KNN.fit(X_train,y_train)

neigh = KNN.kneighbors(X_train, return_distance=False)
# print('Kneighbors:\n',neigh)

#-- Confidence score --------------
cscore = conf_count(neigh,y_train)
# print('Confidence Score:\n',cscore)
xi -= 0.1
Xd = X_train[cscore>=xi]
yd = y_train[cscore>=xi]
# print('Denoised Data (Xd,yd):\n',np.column_stack((Xd,yd)))

Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, train_size=0.7,
                                random_state=42, stratify=yd)


total = len(X)
initial = len(X_train)
denoised = len(Xd_train)

print(f"\nTotal data points: {total}\nInitial training points: {initial}\nDenoised training points: {denoised}")

# Show some successfully denoised data points
denoised_indices = np.where(cscore >= xi)[0]  
print(f"\033[91m\nRandom 3 good samples with confidence score {xi} and above:\033[0m:")
for i in np.random.choice(denoised_indices, 3, replace=False):
    print(f"Data point #{i}; Ground truth: {targets[y[i]]}, Score: {cscore[i]}")


# Show some datapoints that didn't meet the confidence score
ineligible_indices = np.where(cscore < xi)[0]
print(f"\033[91m\nRandom 3 bad samples with confidence score below {xi}:\033[0m:")
for i in np.random.choice(ineligible_indices, 3, replace=False):
    print(f"Data point #{i}; Ground truth: {targets[y[i]]}, Score: {cscore[i]}")
        

# Adopted from Homework 1
def multi_run(clf, X_train, y_train, X_test, y_test, NUM_EPOCHS):
    t0 = time.time(); acc = np.zeros([NUM_EPOCHS, 1])
    for i in range(NUM_EPOCHS):
        clf.fit(X_train, y_train)            
        acc[i] = clf.score(X_test, y_test)
    etime = time.time() - t0
    return np.mean(acc)*100, np.std(acc)*100, etime # accmean,acc_std,etime

#=====================================================================
print('\n====== Comparision: Scikit-learn Classifiers =================')
#=====================================================================
import os;

def helper(dataset_type, X_train, y_train, X_test, y_test):
    NUM_EPOCHS = 50
    acc_max=0; Acc_CLF = np.zeros([len(classifiers),1])
    print(f'\033[91m\nPerformance in {dataset_type} dataset:\033[0m:')
    for k, (name, clf) in enumerate(zip(names, classifiers)):
        accmean, acc_std, etime = multi_run(clf, X_train, y_train, X_test, y_test, NUM_EPOCHS)

        Acc_CLF[k] = accmean
        if accmean>acc_max: acc_max,algname = accmean,name
        print('%s: %s: Acc.(mean,std) = (%.2f,%.2f)%%; E-time= %.5f'
            %(os.path.basename(dataname),name,accmean,acc_std,etime/NUM_EPOCHS))
    print('--------------------------------------------------------------')
    print('Acc: (mean,max) = (%.2f,%.2f)%%; Best = %s'
        %(np.mean(Acc_CLF),acc_max,algname))

helper('original', X_train, y_train, X_test,y_test)
helper('denoised', Xd_train, yd_train, Xd_test, yd_test)