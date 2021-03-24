import numpy as np
from sklearn.manifold import trustworthiness
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import coranking
from coranking.metrics import trustworthiness, continuity

#https://coranking.readthedocs.io/en/latest/

df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#contrast data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,784)
# normalize data
X_contrast = X_contrast.astype('float32') / 255.0 - 0.5
X_train = X_train.astype('float32') / 255.0 - 0.5

#run PCA with n=... principal components
pca = PCA(n_components=3)
princa = pca.fit_transform(X_contrast)

#Q = coranking.coranking_matrix(high_data, low_data)
Q = coranking.coranking_matrix(X_contrast, princa)

trust_pca = trustworthiness(Q, min_k=10, max_k=20)
cont_pca = continuity(Q, min_k=10, max_k=20)
print(trust_pca)
print(cont_pca)