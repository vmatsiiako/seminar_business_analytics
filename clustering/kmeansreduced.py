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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")

#drop fist signs
df.drop(df.loc[df['label']==0].index, inplace=True)
df.drop(df.loc[df['label']==4].index, inplace=True)
df.drop(df.loc[df['label']==12].index, inplace=True)
df.drop(df.loc[df['label']==13].index, inplace=True)
df.drop(df.loc[df['label']==18].index, inplace=True)

X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values

#contrast train data
X_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_contrast)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,784)

# normalize train data
X_contrast = X_contrast.astype('float32') / 255.0 - 0.5
X_train = X_train.astype('float32') / 255.0 - 0.5

#run kmeans with 24 clusters
kmeans = KMeans(init="k-means++", n_clusters=19, n_init=4)

#run k-means on full dataset train
kmeans_full = kmeans.fit(X_contrast)
labels = kmeans.predict(X_contrast)

print('Number of iterations Full Kmeans train data {}'.format(kmeans_full.n_iter_))

#Print scores full train dataset
print('Homogeneity Score Full Train Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Train Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full train Dataset: {}'.format(v_measure_score(y_train, labels)))