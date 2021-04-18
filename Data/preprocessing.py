import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns

#Initialize constants
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
INTRINSIC_DIMENSIONALITY = 13

# load in the data
df_train = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values

#contrast the data
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

[y_train, X_contrast_train].to_csv("preprocessed_training_data.csv")
[y_test, X_contrast_test].to_csv("preprocessed_training_data.csv")
