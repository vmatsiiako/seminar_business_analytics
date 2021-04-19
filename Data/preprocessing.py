import cv2

import pandas as pd
import numpy as np
from constants import MAX_BRIGHTNESS, NUMBER_OF_PIXELS, MEAN


# load in the data
df_train = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values

# Contrast the data
X_contrast_train = np.zeros(np.shape(X_train))
for i in range(len(X_contrast_train)):
    image = X_train[i,:]
    image = image.astype(np.uint8)
    X_contrast_train[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)
    
X_contrast_test = np.zeros(np.shape(X_test))
for i in range(len(X_contrast_test)):
    image = X_test[i,:]
    image = image.astype(np.uint8)
    X_contrast_test[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# normalize train data
X_contrast_train = X_contrast_train.astype('float32') / MAX_BRIGHTNESS - MEAN
X_contrast_test = X_contrast_test.astype('float32') / MAX_BRIGHTNESS - MEAN

# Concatenate the feature dataset with the labels
train_contrast = np.concatenate((y_train.reshape(-1,1), X_contrast_train), axis=1)
test_contrast = np.concatenate((y_test.reshape(-1,1), X_contrast_test), axis=1)

# Save the preprocessed data sets
np.savetxt("train.csv", train_contrast, delimiter=',')
np.savetxt("test.csv", test_contrast, delimiter=',')
