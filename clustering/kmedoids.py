import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn_extra.cluster import KMedoids

#Initialize constants
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784

#load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#load in embedding from deep autoencoders on train data
df_ae_deep = pd.read_csv("../Data/TEST_FINAL_THIS_TIME_DEFINITELY_FINAL_FINAL_FINAL_reduced_trainset_with_noise__BATCH_SIZE_64_P_NOISE_TYPE_zeros_P_NOISE_PERCENTAGE_0,5_LAYERS_[620,330,13]_LR_0,001_EPOCH_9.csv", header=None)
X_train_ae_deep = df_ae_deep.iloc[:,0:].values

#load in embedding from denoised autoencoders on train data
df_ae_denoised = pd.read_csv("../Data/FINAL_DENOISING_reduced_trainset_with_noise__BATCH_SIZE_64_P_NOISE_TYPE_gaussian_P_NOISE_PERCENTAGE_0,1_F_NOISE_TYPE_gaussian_F_NOISE_PERCENTAGE_0,3_LAYERS_[620,330,13]_LR_0,001_EPOCH_32.csv", header=None)
X_train_ae_denoised = df_ae_denoised.iloc[:,0:].values

#load in test data
df_test = pd.read_csv("../Data/sign_mnist_test.csv")[1500:]
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values

#load in embedding from deep autoencoders on test data
df_test_ae_deep = pd.read_csv("../Data/TEST_FINAL_THIS_TIME_DEFINITELY_FINAL_FINAL_FINAL_reduced_test_set_with_noise_BATCH_SIZE_64_P_NOISE_TYPE_zeros_P_NOISE_PERCENTAGE_0,5_LAYERS_[620,330,13]_LR_0,001_EPOCH_9.csv", header=None)
X_test_ae_deep = df_test_ae_deep.iloc[:,0:].values

#load in embedding from denoised autoencoders on test data
df_test_ae_denoised = pd.read_csv("../Data/FINAL_DENOISING_reduced_test_set_with_noise_BATCH_SIZE_64_P_NOISE_TYPE_gaussian_P_NOISE_PERCENTAGE_0,1_F_NOISE_TYPE_gaussian_F_NOISE_PERCENTAGE_0,3_LAYERS_[620,330,13]_LR_0,001_EPOCH_32.csv", header=None)
X_test_ae_denoised = df_ae_denoised.iloc[:,0:].values

#contrast data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,NUMBER_OF_PIXELS)

# normalize data
X_contrast = X_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN

#contrast test data
X_contrast_test = np.zeros(np.shape(X_test))
for i in range(len(X_contrast_test)):
    image = X_test[i,:]
    image = image.astype(np.uint8)
    X_contrast_test[i] = cv2.equalizeHist(image).reshape(1,NUMBER_OF_PIXELS)

# normalize data
X_contrast_test = X_contrast_test.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

#run PCA with n=13 principal components
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)

#run PCA with n=13 principal components on test data
pca = PCA(n_components=13)
princa_test = pca.fit_transform(X_contrast_test)

#initialize K_medoid with 24 clusters
pam = KMedoids(n_clusters = 24, random_state = 0)

#pam on full contrasted data
pam_full = pam.fit(X_contrast)
labels = pam.predict(X_contrast)

#pam on PCA train
pam_pca = pam.fit(princa)
labels_pca = pam.predict(princa)

#pam on deep ae train
pam_ae_deep = pam.fit(X_train_ae_deep)
labels_ae_deep = pam.predict(X_train_ae_deep)

#pam on denoised ae train
pam_ae_denoised = pam.fit(X_train_ae_denoised)
labels_ae_denoised = pam.predict(X_train_ae_denoised)

#pam on full contrasted test data
pam_full_test = pam.fit(X_contrast_test)
labels_test = pam.predict(X_contrast_test)

#pam on PCA on test data
pam_pca_test = pam.fit(princa_test)
labels_pca_test = pam.predict(princa_test)

#pam on deep ae test
pam_ae_test_deep = pam.fit(X_test_ae_deep)
labels_ae_test_deep = pam.predict(X_test_ae_deep)

#pam on denoised ae test
pam_ae_test_denoised = pam.fit(X_test_ae_denoised)
labels_ae_test_denoised = pam.predict(X_test_ae_denoised)

#print number of iterations train dataset
print('Number of iterations Full Kmedoid {}'.format(pam_full.n_iter_))
print('Number of iterations PCA Kmedoid {}'.format(pam_pca.n_iter_))
print('Number of iterations Deep AE Kmedoid {}'.format(pam_ae_deep.n_iter_))
print('Number of iterations Denoised AE Kmedoid {}'.format(pam_ae_denoised.n_iter_))

#print number of iterations test dataset
print('Number of iterations Full Kmedoid {}'.format(pam_full_test.n_iter_))
print('Number of iterations PCA Kmedoid {}'.format(pam_pca_test.n_iter_))
print('Number of iterations Deep AE Kmedoid {}'.format(pam_ae_test_deep.n_iter_))
print('Number of iterations Denoised AE Kmedoid {}'.format(pam_ae_test_denoised.n_iter_))

#Print scores full train dataset
print('Homogeneity Score Full Train Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Train Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full train Dataset: {}'.format(v_measure_score(y_train, labels)))

#Print Scores PCA train data
print('Homogeneity Score PCA Train Dataset: {}'.format(homogeneity_score(y_train, labels_pca)))
print('Completeness Score PCA Train Dataset: {}'.format(completeness_score(y_train, labels_pca)))
print('V-score Score PCA Train Dataset: {}'.format(v_measure_score(y_train, labels_pca)))

#Print Scores deep autoencoders train data
print('Homogeneity Score Deep AE Train Dataset: {}'.format(homogeneity_score(y_train, labels_ae_deep)))
print('Completeness Score Deep AE Train Dataset: {}'.format(completeness_score(y_train, labels_ae_deep)))
print('V-score Score Deep AE Train Dataset: {}'.format(v_measure_score(y_train, labels_ae_deep)))

#Print Scores denoised autoencoders train data
print('Homogeneity Score Denoised AE Train Dataset: {}'.format(homogeneity_score(y_train, labels_ae_denoised)))
print('Completeness Score Denoised AE Train Dataset: {}'.format(completeness_score(y_train, labels_ae_denoised)))
print('V-score Score Denoised AE Train Dataset: {}'.format(v_measure_score(y_train, labels_ae_denoised)))

#Print scores full test dataset
print('Homogeneity Score Full Test Dataset: {}'.format(homogeneity_score(y_test, labels_test)))
print('Completeness Score Full Test Dataset: {}'.format(completeness_score(y_test, labels_test)))
print('V-score Score Full Test Dataset: {}'.format(v_measure_score(y_test, labels_test)))

#Print Scores PCA test data
print('Homogeneity Score Test Dataset PCA: {}'.format(homogeneity_score(y_test, labels_pca_test)))
print('Completeness Score Test Dataset PCA: {}'.format(completeness_score(y_test, labels_pca_test)))
print('V-score Score Test Dataset PCA: {}'.format(v_measure_score(y_test, labels_pca_test)))

#Print Scores autoencoders test data
print('Homogeneity Score Test Dataset Deep AE: {}'.format(homogeneity_score(y_test, labels_ae_test_deep)))
print('Completeness Score Test Dataset Deep AE: {}'.format(completeness_score(y_test, labels_ae_test_deep)))
print('V-score Score Test Dataset Deep AE: {}'.format(v_measure_score(y_test, labels_ae_test_deep)))

#Print Scores autoencoders test data
print('Homogeneity Score Test Dataset Denoised AE: {}'.format(homogeneity_score(y_test, labels_ae_test_denoised)))
print('Completeness Score Test Dataset Denoised Ae: {}'.format(completeness_score(y_test, labels_ae_test_denoised)))
print('V-score Score Test Dataset Denoised AE: {}'.format(v_measure_score(y_test, labels_ae_test_denoised)))