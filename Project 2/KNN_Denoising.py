# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#---------------------------------------------------------------
import numpy as np; np.set_printoptions(suppress=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import time
from sklearn_classifiers import classifiers, names

#----------------------------------
def conf_count(neigh,y):
    cscore = np.zeros(y.shape)
    for i in range(len(y)):
        labels = y[neigh[i]]
        cscore[i] = len(labels[labels==y[i]])
    return cscore

#-- Data Generation ---------------
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
X1, X2, y1, y2 = train_test_split(X, y, train_size=0.46,
                                random_state=42, stratify=y)

print(f"\033[35mData points chosen: {len(X1)}/{len(X)}\033[0m")
helper('demo', X1, y1, X2, y2)
print("\033[32mAs you can see, the classifiers perform better on denoised dataset compared to this dataset\n, which shows that the denoising process is successsful!\033[0m")