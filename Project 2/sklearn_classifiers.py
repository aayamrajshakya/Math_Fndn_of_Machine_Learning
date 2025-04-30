from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

classifiers = [
    LogisticRegression(max_iter=1000),
    KNeighborsClassifier(5),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1)
]

names = [
    "Logistic-Regr",
    "Kneighbors-5",
    "SVC",
    "Random forest",
]