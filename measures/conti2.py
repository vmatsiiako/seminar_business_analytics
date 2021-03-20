import numpy as np
from sklearn.manifold import trustworthiness
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import coranking
from coranking.metrics import continuity
df = pd.read_csv("../Data/sign_mnist_train.csv")

features = df.columns[1:]
X = df.loc[:, features].values
pca = PCA(n_components=5)
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)
princa = pca.fit_transform(x)

Q = coranking.coranking_matrix(df, princa)

trust_pca = trustworthiness(Q, min_k=10, max_k=20)
#cont_pca = continuity(Q, min_k=10, max_k=20)
print(trust_pca)
#print(cont_pca)