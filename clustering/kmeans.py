import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# load in the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
features = df.columns[1:]
X_train = df.loc[:, features].values
y = df.loc[:,['label']].values


sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, n_jobs='deprecated', algorithm='auto').fit(x)