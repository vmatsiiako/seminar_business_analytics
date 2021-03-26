import numpy as np
from sklearn.manifold import trustworthiness
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

trust_array = np.zeros(26)
for i in range(1, 26):
    trust = trustworthiness(X_contrast, princa, n_neighbors=i)
    trust_array[i - 1] = trust


#trust2 = trustworthiness(X_contrast, princa, n_neighbors=10)
# print(trust)
# print(features)
# print(x_train)
#print(trust2)
print(trust_array)
