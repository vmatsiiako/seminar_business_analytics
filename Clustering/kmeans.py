import pandas as pd
# import matplotlib
import numpy as np
import cv2

# matplotlib.use('TkAgg')
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans
from constants import MAX_BRIGHTNESS, NUMBER_OF_PIXELS, MEAN


# Load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:, 1:].values
y_train = df.iloc[:, 0].values

# Load in embedding from deep autoencoders on train data
df_ae_deep = pd.read_csv("../Data/Final_train_ae.csv", header=None)
X_train_ae_deep = df_ae_deep.iloc[:, 0:].values

# Load in embedding from denoised autoencoders on train data
df_ae_denoised = pd.read_csv("../Data/Final_train_denoising_ae.csv", header=None)
X_train_ae_denoised = df_ae_denoised.iloc[:, 0:].values

# load in test data
df_test = pd.read_csv("../Data/sign_mnist_test.csv")[1500:]
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

# load in embedding from deep autoencoders on test data
df_test_ae_deep = pd.read_csv("../Data/Final_test_ae.csv", header=None)
X_test_ae_deep = df_test_ae_deep.iloc[:, 0:].values

# load in embedding from denoised autoencoders on test data
df_test_ae_denoised = pd.read_csv("../Data/Final_test_denoising_ae.csv", header=None)
X_test_ae_denoised = df_test_ae_denoised.iloc[:, 0:].values

# contrast train data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# normalize train data
X_contrast = X_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN

# contrast test data
X_contrast_test = np.zeros(np.shape(X_test))
for i in range(len(X_contrast_test)):
    image = X_test[i, :]
    image = image.astype(np.uint8)
    X_contrast_test[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# normalize test data
X_contrast_test = X_contrast_test.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

# run PCA with n=13 principal components on training set
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)

# run PCA with n=13 principal components on test data
pca = PCA(n_components=13)
princa_test = pca.fit_transform(X_contrast_test)

# run kmeans with 24 clusters
kmeans = KMeans(init="k-means++", n_clusters=24, n_init=4)

# run k-means on full train dataset
kmeans_full = kmeans.fit(X_contrast)
labels = kmeans.predict(X_contrast)

# run kmeans on PCA train
kmeans_pca = kmeans.fit(princa)
labels_pca = kmeans.predict(princa)

# run kmeans on deep autoencoders train
kmeans_ae_deep = kmeans.fit(X_train_ae_deep)
labels_ae_deep = kmeans.predict(X_train_ae_deep)

# run kmeans on denoised autoencoders train
kmeans_ae_denoised = kmeans.fit(X_train_ae_denoised)
labels_ae_denoised = kmeans.predict(X_train_ae_denoised)

# run k-means on full dataset test
kmeans_full_test = kmeans.fit(X_contrast_test)
labels_test = kmeans.predict(X_contrast_test)

# run kmeans on PCA test
kmeans_pca_test = kmeans.fit(princa_test)
labels_pca_test = kmeans.predict(princa_test)

# run kemans on deep autoencoders test
kmeans_ae_test_deep = kmeans.fit(X_test_ae_deep)
labels_ae_test_deep = kmeans.predict(X_test_ae_deep)

# run kemans on denoised autoencoders test
kmeans_ae_test_denoised = kmeans.fit(X_test_ae_denoised)
labels_ae_test_denoised = kmeans.predict(X_test_ae_denoised)

# print number of iterations train data
print('Number of iterations Full Kmeans train data {}'.format(kmeans_full.n_iter_))
print('Number of iterations PCA Kmeans train data {}'.format(kmeans_pca.n_iter_))
print('Number of iterations deep autoencoder Kmeans train data {}'.format(kmeans_ae_deep.n_iter_))
print('Number of iterations denoised autoencoder Kmeans train data {}'.format(kmeans_ae_denoised.n_iter_))

# print number of iterations test data
print('Number of iterations Full Kmeans test data{}'.format(kmeans_full_test.n_iter_))
print('Number of iterations PCA Kmeans test data {}'.format(kmeans_pca_test.n_iter_))
print('Number of iterations deep autoencoder Kmeans test data {}'.format(kmeans_ae_test_deep.n_iter_))
print('Number of iterations denoised autoencoder Kmeans test data {}'.format(kmeans_ae_test_denoised.n_iter_))

# Print scores full train dataset
print('Homogeneity Score Full Train Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Train Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full train Dataset: {}'.format(v_measure_score(y_train, labels)))

# Print Scores PCA train data
print('Homogeneity Score PCA Train Dataset: {}'.format(homogeneity_score(y_train, labels_pca)))
print('Completeness Score PCA Train Dataset: {}'.format(completeness_score(y_train, labels_pca)))
print('V-score Score PCA Train Dataset: {}'.format(v_measure_score(y_train, labels_pca)))

# Print Scores deep autoencoders train data
print('Homogeneity Score Deep AE Train Dataset: {}'.format(homogeneity_score(y_train, labels_ae_deep)))
print('Completeness Score Deep AE Train Dataset: {}'.format(completeness_score(y_train, labels_ae_deep)))
print('V-score Score Deep AE Train Dataset: {}'.format(v_measure_score(y_train, labels_ae_deep)))

# Print Scores deep denoised autoencoders train data
print('Homogeneity Score Denoised AE Train Dataset: {}'.format(homogeneity_score(y_train, labels_ae_denoised)))
print('Completeness Score Denoised AE Train Dataset: {}'.format(completeness_score(y_train, labels_ae_denoised)))
print('V-score Score Denoised AE Train Dataset: {}'.format(v_measure_score(y_train, labels_ae_denoised)))

# Print scores full test dataset
print('Homogeneity Score Full Test Dataset: {}'.format(homogeneity_score(y_test, labels_test)))
print('Completeness Score Full Test Dataset: {}'.format(completeness_score(y_test, labels_test)))
print('V-score Score Full Test Dataset: {}'.format(v_measure_score(y_test, labels_test)))

# Print Scores PCA test data
print('Homogeneity Score Test Dataset PCA: {}'.format(homogeneity_score(y_test, labels_pca_test)))
print('Completeness Score Test Dataset PCA: {}'.format(completeness_score(y_test, labels_pca_test)))
print('V-score Score Test Dataset PCA: {}'.format(v_measure_score(y_test, labels_pca_test)))

# Print Scores deep autoencoders test data
print('Homogeneity Score Test Dataset Deep AE: {}'.format(homogeneity_score(y_test, labels_ae_test_deep)))
print('Completeness Score Test Dataset Deep AE: {}'.format(completeness_score(y_test, labels_ae_test_deep)))
print('V-score Score Test Dataset Deep AE: {}'.format(v_measure_score(y_test, labels_ae_test_deep)))

# Print Scores denoised autoencoders test data
print('Homogeneity Score Test Dataset Denpoised AE: {}'.format(homogeneity_score(y_test, labels_ae_test_denoised)))
print('Completeness Score Test Dataset Denoised AE: {}'.format(completeness_score(y_test, labels_ae_test_denoised)))
print('V-score Score Test Dataset Denoised AE: {}'.format(v_measure_score(y_test, labels_ae_test_denoised)))
