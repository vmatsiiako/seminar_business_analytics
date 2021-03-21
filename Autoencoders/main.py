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


# CONSTANTS
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
BATCH_SIZE = [32, 64]
NOISE_TYPE = 'zeros'
NOISE_PERCENTAGE = [0.1, 0.2]  #set it to "None" to impose gaussian noise
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



# acv = GridSearchCV(model, parameters)
# acv.fit(X=train_dl_clean, train_dl_clean=train_dl_clean, train_dl_gaussian=train_dl_gaussian, train_dl_zeros=train_dl_zeros)


for i in range(1):
    # noise_percentage = random.sample(NOISE_PERCENTAGE, 1)[0],
    # batch_size = random.sample(BATCH_SIZE, 1),
    noise_percentage = NOISE_PERCENTAGE[0]
    batch_size = BATCH_SIZE[0]
    # increase the contract of pictures
    X_contrast = np.zeros(np.shape(X_train))
    for i in range(len(X_contrast)):
        image = X_train[i, :]
        image = image.astype(np.uint8)
        X_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

    # normalize data
    X_contrast = X_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
    X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN
    X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

    # Convert the data to torch types
    X_clean = torch.Tensor(X_contrast)
    X_noise = np.zeros(np.shape(X_contrast))
    for i in range(len(X_contrast)):
        X_noise[i] = add_noise(X_contrast[i, :], noise_type=NOISE_TYPE, percentage=noise_percentage)
    X_noise = torch.Tensor(X_noise)

    train_ds_clean = TensorDataset(X_clean)
    train_ds_noise = TensorDataset(X_noise)
    train_dl_clean = DataLoader(train_ds_clean, batch_size=batch_size, shuffle=False)
    train_dl_noise = DataLoader(train_ds_noise, batch_size=batch_size, shuffle=False)

    model = Model()
    model.fit(noise_percentage,
              batch_size,
              HIDDEN_LAYERS[0],
              train_dl_clean,
              train_dl_noise)



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