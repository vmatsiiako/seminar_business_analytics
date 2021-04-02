import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2

#define set of constants
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28

#load in train data
df = pd.read_csv("../Data/sign_mnist_train.csv")
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

#perform PCA on the contrasted data
pca = PCA(n_components=13)
principalComponents = pca.fit_transform(X_contrast)

#approximate high-dimensional data with the embedding
full_approx = pca.inverse_transform(principalComponents)

#start creating the images
plt.figure(figsize=(8,4))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(X_contrast[3].reshape(28,28),
           cmap = plt.cm.gray)
plt.xlabel('784 components', fontsize = 14)
plt.title('Original Image', fontsize = 20)

# Plot the image based on the approximation
plt.subplot(1, 2, 2);
plt.imshow(full_approx[3].reshape(28, 28),
              cmap = plt.cm.gray);
plt.xlabel('13 components', fontsize = 14)
plt.title('Reconstructed Image', fontsize = 20)
plt.show()