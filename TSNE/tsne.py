import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2

# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
df = pd.read_csv("../Data/sign_mnist_train.csv")

features = df.columns[1:]
X_train = df.loc[:, features].values

# increase the contract of pictures
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1, 784)

pca = PCA(n_components=13)
# x = df.loc[:, features].values
x = StandardScaler().fit_transform(X_contrast)
princa = pca.fit_transform(x)

#run tsne
TSNE = TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None, square_distances='legacy')