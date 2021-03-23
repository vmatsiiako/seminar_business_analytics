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
NOISE_TYPE = ['zeros', 'gaussian']
NOISE_PERCENTAGE = [0, 0.1, 0.2]  #set it to "None" to impose gaussian noise
GAUSSIAN_ST_DEV = None   #set it to "None" to impose zero noise
HIDDEN_LAYERS = [[500, 250, 100, 5], [500, 250, 5]]
EPOCHS_PRETRAINING = 2
EPOCHS_FINETUNING = 2
NUMBER_FOLDS = 5


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

kf = KFold(n_splits=NUMBER_FOLDS)

for i in range(1):
    # noise_percentage = random.sample(NOISE_PERCENTAGE, 1)[0],
    # batch_size = random.sample(BATCH_SIZE, 1),
    noise_percentage = random.sample(NOISE_PERCENTAGE, 1)[0]
    noise_type = random.sample(NOISE_TYPE, 1)[0]
    batch_size = random.sample(BATCH_SIZE, 1)[0]
    hidden_layers = random.sample(HIDDEN_LAYERS, 1)[0]

    current_validation_losses = np.zeros((EPOCHS_FINETUNING,NUMBER_FOLDS))
    current_training_losses = np.zeros((EPOCHS_FINETUNING, NUMBER_FOLDS))
    column = 0
    optimal_loss = float('inf')
    for train_index, test_index in kf.split(X_train_contrast):
        X_train_CV, X_validation_CV = X_train_contrast[train_index], X_train_contrast[test_index]
        # Convert the data to torch types
        X_train_clean = torch.Tensor(X_train_CV)
        X_validation_clean = torch.Tensor(X_validation_CV)
        X_train_noise = np.zeros(np.shape(X_train_CV))
        for i in range(len(X_train_CV)):
            X_train_noise[i] = add_noise(X_train_CV[i, :], noise_type=NOISE_TYPE, percentage=noise_percentage)
        X_train_noise = torch.Tensor(X_train_noise)

        train_ds_clean = TensorDataset(X_train_clean)
        train_ds_noise = TensorDataset(X_train_noise)
        validation_ds = TensorDataset(X_validation_clean)
        train_dl_clean = DataLoader(train_ds_clean, batch_size=batch_size, shuffle=False)
        train_dl_noise = DataLoader(train_ds_noise, batch_size=batch_size, shuffle=False)
        validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

        model = Model()
        val_loss, train_loss, ae = model.fit(noise_percentage,
                                             batch_size,
                                             hidden_layers,
                                             train_dl_clean,
                                             train_dl_noise,
                                             validation_dl)
        val_loss = np.array(val_loss)
        train_loss = np.array(train_loss)
        current_validation_losses[:, column] = val_loss
        current_training_losses[:, column] = train_loss
        column += 1

    average_validation_loss = current_validation_losses.mean(axis=1)
    average_training_loss = current_training_losses.mean(axis=1)
    minimum_loss = average_validation_loss.min()
    epoch = average_validation_loss.argmin()+1

    N = np.arange(0, EPOCHS_FINETUNING)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, average_training_loss, label="train_loss")
    plt.plot(N, average_validation_loss, label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(f"loss_graph"
                f"_BATCH_SIZE_{str(batch_size)}"
                f"_NOISE_TYPE_{noise_type}"
                f"_NOISE_PERCENTAGE_{str(noise_percentage)}"
                f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in hidden_layers])}]")

    if i == 0 or minimum_loss < optimal_loss:
        optimal_loss = minimum_loss
        optimal_noise = noise_percentage
        optimal_batch_size = batch_size
        optimal_hidden_layers = hidden_layers

X_test_contrast = torch.Tensor(X_test_contrast)
test_ds = TensorDataset(X_test_contrast)
visualize = DataLoader(test_ds, batch_size=1, shuffle=False)
final_model = Model()
test_loss, training, autoencoder = final_model.fit(optimal_noise,
                                         optimal_batch_size,
                                         optimal_hidden_layers,
                                         train_dl_clean,
                                         train_dl_noise,
                                         visualize)

X_test_contrast = torch.Tensor(X_test_contrast)
test_ds = TensorDataset(X_test_contrast)
visualize = DataLoader(test_ds, batch_size=1, shuffle=False)
NUMBER_OF_PICTURES_TO_DISPLAY = 10  # How many pictures we will display
plt.figure(figsize=(20, 4))
for i, features in enumerate(visualize):
    # Display original
    ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1)
    plt.imshow(features[0].numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1 + NUMBER_OF_PICTURES_TO_DISPLAY)
    plt.imshow(autoencoder(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 9:
        break

plt.savefig('final_test_prediction')