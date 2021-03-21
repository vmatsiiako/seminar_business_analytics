import random

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Autoencoders.DAE import DAE
from Autoencoders.d_DAE import d_DAE
from Autoencoders.utils import add_noise
from Autoencoders.model import Model
from sklearn.model_selection import KFold


# CONSTANTS
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
BATCH_SIZE = [16, 32, 64]
NOISE_TYPE = 'zeros'
NOISE_PERCENTAGE = [0, 0.1, 0.2]  #set it to "None" to impose gaussian noise
GAUSSIAN_ST_DEV = None   #set it to "None" to impose zero noise
HIDDEN_LAYERS = [[500, 250, 100, 5], [500, 250, 5]]
EPOCHS_PRETRAINING = 10
EPOCHS_FINETUNING = 10

parameters = {"NOISE_PERCENTAGE": NOISE_PERCENTAGE
    # ,
    #           'hidden_layers': HIDDEN_LAYERS
              }

# import and extract the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_train = df.iloc[:,1:].values 
y_train = df.iloc[:,0].values
X_test = df_test.iloc[:,1:].values 
y_test = df_test.iloc[:,0].values

X_train_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_train_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_train_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

X_test_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_test_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_test_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# normalize data
X_train_contrast = X_train_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test_contrast = X_test_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

# acv = GridSearchCV(model, parameters)
# acv.fit(X=train_dl_clean, train_dl_clean=train_dl_clean, train_dl_gaussian=train_dl_gaussian, train_dl_zeros=train_dl_zeros)


for i in range(4):
    # noise_percentage = random.sample(NOISE_PERCENTAGE, 1)[0],
    # batch_size = random.sample(BATCH_SIZE, 1),
    noise_percentage = random.sample(NOISE_PERCENTAGE, 1)[0]
    batch_size = random.sample(BATCH_SIZE, 1)[0]
    hidden_layers = random.sample(HIDDEN_LAYERS, 1)[0]
    # increase the contract of pictures
    # X_train_contrast = np.zeros(np.shape(X_train))
    # for i in range(len(X_train_contrast)):
    #     image = X_train[i, :]
    #     image = image.astype(np.uint8)
    #     X_train_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)
    #
    # X_test_contrast = np.zeros(np.shape(X_train))
    # for i in range(len(X_test_contrast)):
    #     image = X_train[i, :]
    #     image = image.astype(np.uint8)
    #     X_test_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

    # # normalize data
    # X_train_contrast = X_train_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
    # X_test_contrast = X_test_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
    # X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN
    # X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X_train_contrast):
        X_train_CV, X_validation_CV = X_train_contrast[train_index], X_train_contrast[test_index]
        # Convert the data to torch types
        X_train_clean = torch.Tensor(X_train_contrast)
        X_test_clean = torch.Tensor(X_test_contrast)
        X_train_noise = np.zeros(np.shape(X_train_contrast))
        for i in range(len(X_train_contrast)):
            X_train_noise[i] = add_noise(X_train_contrast[i, :], noise_type=NOISE_TYPE, percentage=noise_percentage)
        X_train_noise = torch.Tensor(X_train_noise)

        train_ds_clean = TensorDataset(X_train_clean)
        train_ds_noise = TensorDataset(X_train_noise)
        test_ds = TensorDataset(X_test_clean)
        train_dl_clean = DataLoader(train_ds_clean, batch_size=batch_size, shuffle=False)
        train_dl_noise = DataLoader(train_ds_noise, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # model = Model()
        # model.fit(noise_percentage,
        #           batch_size,
        #           HIDDEN_LAYERS[0],
        #           train_dl_clean,
        #           train_dl_noise,
        #           test_dl)



# X_test = torch.Tensor(X_test)
# test_ds = TensorDataset(X_test)
# visualize = DataLoader(test_ds, batch_size=1, shuffle=False)
# NUMBER_OF_PICTURES_TO_DISPLAY = 10  # How many pictures we will display
# plt.figure(figsize=(20, 4))
# for i, features in enumerate(visualize):
#     # Display original
#     ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1)
#     plt.imshow(features[0].numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # Display reconstruction
#     ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1 + NUMBER_OF_PICTURES_TO_DISPLAY)
#     plt.imshow(dae(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     if i == 9:
#         break