import pandas as pd
import matplotlib
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
import cv2
matplotlib.use('TkAgg')
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans

#Initialize constants
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784

#load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#load in embedding from autoencoders on train data
#df_ae = pd.read_csv("../Data/reduced_trainset_3.csv", header=None)
#df_ae = pd.read_csv("../Data/reduced_trainset_with_BATCH_SIZE_64_NOISE_TYPE_gaussian_NOISE_PERCENTAGE_2_HIDDEN_LAYERS_[620,330,100,13]_LEATNING_RATE_0,002_EPOCH_70.csv", header=None)
df_ae = pd.read_csv("../Data/reduced_trainset_with_noise__BATCH_SIZE_32_P_NOISE_TYPE_gaussian_P_NOISE_PERCENTAGE_2_F_NOISE_TYPE_zeros_F_NOISE_PERCENTAGE_0,2_LAYERS_[620,330,13]_LR_0,002_EPOCH_20.csv", header=None)
X_train_ae = df_ae.iloc[:,0:].values

#load in test data
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values

#load in embedding from autoencoders on test data
#df_test_ae = pd.read_csv("../Data/reduced_testset_3.csv", header=None)
#df_test_ae = pd.read_csv("../Data/reduced_test_set_with_BATCH_SIZE_64_NOISE_TYPE_gaussian_NOISE_PERCENTAGE_2_HIDDEN_LAYERS_[620,330,100,13]_LEATNING_RATE_0,002_EPOCH_70.csv", header=None)
df_test_ae = pd.read_csv("../Data/reduced_test_set_with_noise_BATCH_SIZE_32_P_NOISE_TYPE_gaussian_P_NOISE_PERCENTAGE_2_F_NOISE_TYPE_zeros_F_NOISE_PERCENTAGE_0,2_LAYERS_[620,330,13]_LR_0,002_EPOCH_20.csv", header=None)
X_test_ae = df_test_ae.iloc[:,0:].values

#contrast train data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# normalize train data
X_contrast = X_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN

#contrast test data
X_contrast_test = np.zeros(np.shape(X_test))
for i in range(len(X_contrast_test)):
    image = X_test[i,:]
    image = image.astype(np.uint8)
    X_contrast_test[i] = cv2.equalizeHist(image).reshape(1,784)

# normalize test data
X_contrast_test = X_contrast_test.astype('float32') / 255.0 - 0.5
X_test = X_test.astype('float32') / 255.0 - 0.5

#run PCA with n=13 principal components on training set
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

#run kmeans on autoencoders train
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

#print number of iterations train data
print('Number of iterations Full Kmeans train data {}'.format(kmeans_full.n_iter_))
print('Number of iterations PCA Kmeans train data {}'.format(kmeans_pca.n_iter_))
print('Number of iterations autoencoder Kmeans train data {}'.format(kmeans_ae.n_iter_))

#print number of iterations test data
print('Number of iterations Full Kmeans test data{}'.format(kmeans_full_test.n_iter_))
print('Number of iterations PCA Kmeans test data {}'.format(kmeans_pca_test.n_iter_))
print('Number of iterations autoencoder Kmeans test data {}'.format(kmeans_ae_test.n_iter_))

#Print scores full train dataset
print('Homogeneity Score Full Train Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Train Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full train Dataset: {}'.format(v_measure_score(y_train, labels)))

#Print Scores PCA train data
print('Homogeneity Score PCA Train Dataset: {}'.format(homogeneity_score(y_train, labels_pca)))
print('Completeness Score PCA Train Dataset: {}'.format(completeness_score(y_train, labels_pca)))
print('V-score Score PCA Train Dataset: {}'.format(v_measure_score(y_train, labels_pca)))

#Print Scores autoencoders train data
print('Homogeneity Score AE Train Dataset: {}'.format(homogeneity_score(y_train, labels_ae)))
print('Completeness Score AE Train Dataset: {}'.format(completeness_score(y_train, labels_ae)))
print('V-score Score AE Train Dataset: {}'.format(v_measure_score(y_train, labels_ae)))

#Print scores full test dataset
print('Homogeneity Score Full Test Dataset: {}'.format(homogeneity_score(y_test, labels_test)))
print('Completeness Score Full Test Dataset: {}'.format(completeness_score(y_test, labels_test)))
print('V-score Score Full Test Dataset: {}'.format(v_measure_score(y_test, labels_test)))

#Print Scores PCA test data
print('Homogeneity Score Test Dataset PCA: {}'.format(homogeneity_score(y_test, labels_pca_test)))
print('Completeness Score Test Dataset PCA: {}'.format(completeness_score(y_test, labels_pca_test)))
print('V-score Score Test Dataset PCA: {}'.format(v_measure_score(y_test, labels_pca_test)))

#Print Scores autoencoders test data
print('Homogeneity Score Test Dataset AE: {}'.format(homogeneity_score(y_test, labels_ae_test)))
print('Completeness Score Test Dataset Ae: {}'.format(completeness_score(y_test, labels_ae_test)))
print('V-score Score Test Dataset AE: {}'.format(v_measure_score(y_test, labels_ae_test)))