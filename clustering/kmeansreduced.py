import pandas
import pandas as pd
import matplotlib
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
import cv2
matplotlib.use('TkAgg')
import numpy as np
from sklearn.cluster import KMeans

MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28

#load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")

#drop fist signs from the dataframe
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
    X_contrast[i] = cv2.equalizeHist(image).reshape(1,NUMBER_OF_PIXELS)

# normalize train data
X_contrast = X_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN

#run kmeans with 19 clusters, as there are 19 letters left in the data
kmeans = KMeans(init="k-means++", n_clusters=19, n_init=4)

#run k-means on full dataset train
kmeans_full = kmeans.fit(X_contrast)
labels = kmeans.predict(X_contrast)

print('Number of iterations Full Kmeans train data {}'.format(kmeans_full.n_iter_))

#Print scores full train dataset
print('Homogeneity Score Full Train Dataset: {}'.format(homogeneity_score(y_train, labels)))
print('Completeness Score Full Train Dataset: {}'.format(completeness_score(y_train, labels)))
print('V-score Score Full train Dataset: {}'.format(v_measure_score(y_train, labels)))