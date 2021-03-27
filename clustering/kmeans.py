import pandas
import pandas as pd
import matplotlib
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
import cv2
matplotlib.use('TkAgg')
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#https://towardsdatascience.com/explaining-k-means-clustering-5298dc47bad6
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#load in data
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

#run PCA with n=13 principal components
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)

#run kmeans with 24 clusters
kmeans = KMeans(init="k-means++", n_clusters=24, n_init=4)

#run k-means on full dataset
kmeans_full = kmeans.fit(X_contrast)
labels = kmeans.predict(X_contrast)

#run kmeans on PCA
kmeans_pca = kmeans.fit(princa)
labels_pca = kmeans.predict(princa)

#Print scores full dataset
print('Homogeneity Score Full Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full Dataset: {}'.format(v_measure_score(y_train, labels)))

#Print Scores PCA data
print('Homogeneity Score PCA: {}'.format(homogeneity_score(y_train, labels_pca)))
print('Completeness Score PCA: {}'.format(completeness_score(y_train, labels_pca)))
print('V-score Score PCA: {}'.format(v_measure_score(y_train, labels_pca)))