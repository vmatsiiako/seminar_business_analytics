import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn_extra.cluster import KMedoids
from sklearn import datasets

# load in the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#contrast data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,784)

X_contrast = X_contrast.astype('float32') / 255.0 - 0.5

pam = KMedoids(n_clusters = 24, random_state = 0)
pam_full = pam.fit(X_contrast)
labels = pam.predict(X_contrast)

#silhouette score
print('PAM Scaled Silhouette Score: {}'.format(silhouette_score(X_contrast, pam_full.labels_, metric='euclidean')))

#measures
homo = homogeneity_score(y_train, labels)
print(homo)
comp = completeness_score(y_train, labels)
print(comp)
v = v_measure_score(y_train, labels)
print(v)