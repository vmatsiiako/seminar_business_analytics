import numpy as np
from sklearn.manifold import trustworthiness
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import distance
from trustcont import *

#https://github.com/isido/dimred/blob/master/dimred/trustcont.py
df = pd.read_csv("../Data/sign_mnist_train.csv")

features = df.columns[1:]
X = df.loc[:, features].values
pca = PCA(n_components=5)
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)
princa = pca.fit_transform(x)

ks = vector_row = np.array([1])

cont = continuity(x, princa, ks)
print(cont)