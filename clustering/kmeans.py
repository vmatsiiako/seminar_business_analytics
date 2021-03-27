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

#load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

df_ae = pd.read_csv("../Data/reduced_trainset_2.csv", header=None)
X_train_ae = df_ae.iloc[:,0:].values

#load in test data
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values

df_test_ae = pd.read_csv("../Data/reduced_testset_2.csv", header=None)
X_test_ae = df_test_ae.iloc[:,0:].values

#contrast data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,784)

# normalize data
X_contrast = X_contrast.astype('float32') / 255.0 - 0.5
X_train = X_train.astype('float32') / 255.0 - 0.5

#contrast test data
X_contrast_test = np.zeros(np.shape(X_test))
for i in range(len(X_contrast_test)):
    image = X_test[i,:]
    image = image.astype(np.uint8)
    X_contrast_test[i] = cv2.equalizeHist(image).reshape(1,784)

# normalize data
X_contrast_test = X_contrast_test.astype('float32') / 255.0 - 0.5
X_test = X_test.astype('float32') / 255.0 - 0.5

#run PCA with n=13 principal components
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)

#run PCA with n=13 principal components on test data
pca = PCA(n_components=13)
princa_test = pca.fit_transform(X_contrast_test)

#run kmeans with 24 clusters
kmeans = KMeans(init="k-means++", n_clusters=24, n_init=4)

#run k-means on full dataset train
kmeans_full = kmeans.fit(X_contrast)
labels = kmeans.predict(X_contrast)

#run kmeans on PCA train
kmeans_pca = kmeans.fit(princa)
labels_pca = kmeans.predict(princa)

#run kemans on autoencoders train
kmeans_ae = kmeans.fit(X_train_ae)
labels_ae = kmeans.predict(X_train_ae)

#run k-means on full dataset test
kmeans_full_test = kmeans.fit(X_contrast_test)
labels_test = kmeans.predict(X_contrast_test)

#run kmeans on PCA test
kmeans_pca_test = kmeans.fit(princa_test)
labels_pca_test = kmeans.predict(princa_test)

#run kemans on autoencoders test
kmeans_ae_test = kmeans.fit(X_test_ae)
labels_ae_test = kmeans.predict(X_test_ae)

#printnumber of iterations
print('Number of iterations Full Kmeans {}'.format(kmeans_full.n_iter_))
print('Number of iterations PCA Kmeans {}'.format(kmeans_pca.n_iter_))
print('Number of iterations autoencoder Kmeans {}'.format(kmeans_ae.n_iter_))

#Print scores full dataset
print('Homogeneity Score Full Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full Dataset: {}'.format(v_measure_score(y_train, labels)))

#Print Scores PCA data
print('Homogeneity Score PCA: {}'.format(homogeneity_score(y_train, labels_pca)))
print('Completeness Score PCA: {}'.format(completeness_score(y_train, labels_pca)))
print('V-score Score PCA: {}'.format(v_measure_score(y_train, labels_pca)))

#Print Scores autoencoders data
print('Homogeneity Score AE: {}'.format(homogeneity_score(y_train, labels_ae)))
print('Completeness Score Ae: {}'.format(completeness_score(y_train, labels_ae)))
print('V-score Score AE: {}'.format(v_measure_score(y_train, labels_ae)))

#Print scores full test dataset
print('Homogeneity Score Test Dataset: {}'.format(homogeneity_score(y_test, labels_test)))
print('Completeness Score Full Dataset: {}'.format(completeness_score(y_test, labels_test)))
print('V-score Score Full Dataset: {}'.format(v_measure_score(y_test, labels_test)))

#Print Scores PCA test data
print('Homogeneity Score Test PCA: {}'.format(homogeneity_score(y_test, labels_pca_test)))
print('Completeness Score Test PCA: {}'.format(completeness_score(y_test, labels_pca_test)))
print('V-score Score Test PCA: {}'.format(v_measure_score(y_test, labels_pca_test)))

#Print Scores autoencoders test data
print('Homogeneity Score Test AE: {}'.format(homogeneity_score(y_test, labels_ae_test)))
print('Completeness Score Test Ae: {}'.format(completeness_score(y_test, labels_ae_test)))
print('V-score Score Test AE: {}'.format(v_measure_score(y_test, labels_ae_test)))