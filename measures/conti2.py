import numpy as np
from sklearn.manifold import trustworthiness
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import coranking
from coranking.metrics import continuity
df = pd.read_csv("../Data/sign_mnist_train.csv")

#https://coranking.readthedocs.io/en/latest/

features = df.columns[1:]
#X = df.loc[:, features].values
pca = PCA(n_components=3)
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)
princa = pca.fit_transform(x)

df2 = pd.read_csv("../Data/sign_mnist_train.csv")

features = df2.columns[1:]
#X2 = df2.loc[:, features].values
pca2 = PCA(n_components=10)
x2 = df2.loc[:, features].values
x2 = StandardScaler().fit_transform(x2)
princa2 = pca2.fit_transform(x2)

Q = coranking.coranking_matrix(princa2, princa)

trust_pca = trustworthiness(Q, min_k=10, max_k=20)
cont_pca = continuity(Q, min_k=10, max_k=20)
print(trust_pca)
print(cont_pca)