import numpy as np
import cv2
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import coranking
import matplotlib.pyplot as plt
from coranking.metrics import trustworthiness, continuity

#load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#load in embedding from autoencoders on train data
df_ae = pd.read_csv("../Data/reduced_trainset_2.csv", header=None)
X_train_ae = df_ae.iloc[:,0:].values

#load in test data
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values

#load in embedding from autoencoders on test data
df_test_ae = pd.read_csv("../Data/reduced_testset_2.csv", header=None)
X_test_ae = df_test_ae.iloc[:,0:].values

#contrast train data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,784)

# normalize train data
X_contrast = X_contrast.astype('float32') / 255.0 - 0.5
X_train = X_train.astype('float32') / 255.0 - 0.5

#contrast test data
X_contrast_test = np.zeros(np.shape(X_test))
for i in range(len(X_contrast_test)):
    image = X_test[i,:]
    image = image.astype(np.uint8)
    X_contrast_test[i] = cv2.equalizeHist(image).reshape(1,784)

# normalize test data
X_contrast_test = X_contrast_test.astype('float32') / 255.0 - 0.5
X_test = X_test.astype('float32') / 255.0 - 0.5

#run PCA with n=13 principal components
pca = PCA(n_components=13)
princa = pca.fit_transform(X_contrast)
princa_test = pca.fit_transform(X_contrast_test)

#run tsne on full data
#TSNE = TSNE(n_components=2, perplexity=40)
#TSNE_output = TSNE.fit_transform(X_contrast)

#pick random subsample to calculate the measures for for the train data
new_data = np.hstack((X_contrast, princa))
new_data_ae = np.hstack((X_contrast, X_train_ae))
n_train = new_data.shape[0]
random_indices = np.random.choice(n_train, size=100, replace=False)
random_sample = new_data[random_indices, :]
full_random = random_sample[:,13:]
pca_random = random_sample[:,:12]
random_ae_sample = new_data_ae[random_indices, :]
ae_random = random_ae_sample[:, :12]

#pick random subsample to calculate the measures for for TSNE
# new_data = np.hstack((X_contrast, TSNE_output))
# number_of_rows = new_data.shape[0]
# random_indices = np.random.choice(number_of_rows, size=100, replace=False)
# random_sample = new_data[random_indices, :]
# full_random_tsne = random_sample[:,2:]
# tsne_random = random_sample[:,:1]

#Q = coranking.coranking_matrix(high_data, low_data)
#calculate coranking matrices
Q = coranking.coranking_matrix(full_random, pca_random)
Q_test = coranking.coranking_matrix(X_contrast_test, princa_test)
Q_ae = coranking.coranking_matrix(full_random, ae_random)
Q_ae_test = coranking.coranking_matrix(X_contrast_test, X_test_ae)

#Q_tsne = coranking.coranking_matrix(full_random_tsne, tsne_random)

#measures for PCA on train data
trust_pca = trustworthiness(Q, min_k=1, max_k=25)
cont_pca = continuity(Q, min_k=1, max_k=25)
print(trust_pca)
print(cont_pca)

#measures for pca on test data
trust_pca_test = trustworthiness(Q_test, min_k = 1, max_k = 25)
cont_pca_test = continuity(Q_test, min_k=1, max_k=25)
print(trust_pca_test)
print(cont_pca_test)

#measures for ae on train data
trust_ae = trustworthiness(Q_ae, min_k=1, max_k=25)
cont_ae = continuity(Q_ae, min_k=1, max_k=25)
print(trust_ae)
print(cont_ae)

#measures for ae on test data
trust_ae_test = trustworthiness(Q_ae_test, min_k=1, max_k=25)
cont_ae_test = continuity(Q_ae_test, min_k=1, max_k=25)
print(trust_ae_test)
print(cont_ae_test)

#measures for tsne
# trust_tsne = trustworthiness(Q_tsne, min_k=1, max_k=25)
# cont_tsne = continuity(Q_tsne, min_k=1, max_k=25)
# print(trust_tsne)
# print(cont_tsne)

#plotting pca on train data
plt.plot(cont_pca, "-m", label="continuity measure")
plt.plot(trust_pca, "-c", label="trustworthiness measure")
plt.legend(loc="upper right")
plt.xlabel('Number of Neighbors')
plt.ylabel('Measure')
plt.set_window_title('pca train data')
plt.show()

#plotting pca on test data
plt.plot(cont_pca_test, "-m", label="continuity measure")
plt.plot(trust_pca_test, "-c", label="trustworthiness measure")
plt.legend(loc="upper right")
plt.xlabel('Number of Neighbors')
plt.ylabel('Measure')
plt.set_window_title('pca test data')
plt.show()

#plotting pca on test data
plt.plot(cont_ae, "-m", label="continuity measure")
plt.plot(trust_ae, "-c", label="trustworthiness measure")
plt.legend(loc="upper right")
plt.xlabel('Number of Neighbors')
plt.ylabel('Measure')
plt.set_window_title('pca test data')
plt.show()

#plotting pca on test data
plt.plot(cont_ae_test, "-m", label="continuity measure")
plt.plot(trust_ae_test, "-c", label="trustworthiness measure")
plt.legend(loc="upper right")
plt.xlabel('Number of Neighbors')
plt.ylabel('Measure')
plt.set_window_title('pca test data')
plt.show()

#plotting tsne
# plt.plot(cont_tsne, "-m", label="continuity measure")
# plt.plot(trust_tsne, "-c", label="trustworthiness measure")
# plt.legend(loc="upper right")
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Measure')
# plt.set_window_title('Tsne')
# plt.show()