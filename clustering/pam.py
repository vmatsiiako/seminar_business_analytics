import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn_extra.cluster import KMedoids

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

#normalize data
X_contrast = X_contrast.astype('float32') / 255.0 - 0.5

#run PCA with n=13 principal components
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)

#initialize K_medoid with 24 clusters
pam = KMedoids(n_clusters = 24, random_state = 0)

#pam on full contrasted data
pam_full = pam.fit(X_contrast)
labels = pam.predict(X_contrast)

#pam on PCA
pam_pca = pam.fit(princa)
labels_pca = pam.predict(princa)

#printnumber of iterations
print('Number of iterations Full Kmeans {}'.format(pam_full.n_iter_))
print('Number of iterations PCA Kmeans {}'.format(pam_pca.n_iter_))

#Print scores full dataset
print('Homogeneity Score Full Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full Dataset: {}'.format(v_measure_score(y_train, labels)))

#Print Scores PCA data
print('Homogeneity Score PCA: {}'.format(homogeneity_score(y_train, labels_pca)))
print('Completeness Score PCA: {}'.format(completeness_score(y_train, labels_pca)))
print('V-score Score PCA: {}'.format(v_measure_score(y_train, labels_pca)))