import numpy as np
from sklearn.manifold import trustworthiness
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/anoukveltman/Downloads/archive/sign_mnist_train.csv")

features = df.columns[1:]
X_train = df.loc[:, features].values

pca = PCA(n_components=5)
#x = df.loc[:, features].values
x = StandardScaler().fit_transform(X_train)
princa = pca.fit_transform(x)

trust = trustworthiness(X_train, princa, n_neighbors=10)
#print(trust)
#print(features)
#print(x_train)
print(X_train)
