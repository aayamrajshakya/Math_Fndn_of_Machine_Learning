# REFERENCE: https://www.geeksforgeeks.org/kmeans-clustering-and-pca-on-wine-dataset/

import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA = datasets.load_wine(as_frame=True)
X = DATA.data
y = DATA.target

# Scale features only
scaler = StandardScaler()
features = scaler.fit_transform(X)

# Minimize the dataset from 15 features to 2 features
pca = PCA(n_components=2)
reduced_X = pd.DataFrame(data=pca.fit_transform(features))

# Stack the reduced dataset with the target classes
NEW_DATA = np.column_stack((reduced_X, y))
np.savetxt("wine.data", NEW_DATA, delimiter=',', fmt="%.3f")