import random

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Autoencoders.utils import add_noise
from Autoencoders.model import Model
from sklearn.model_selection import KFold
import pickle


# CONSTANTS
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
EPOCHS_PRETRAINING = 20
EPOCHS_FINETUNING = 50
NUMBER_FOLDS = 5
optimal_batch_size = 64
optimal_noise_type = 'gaussian'
optimal_noise = 1
optimal_hidden_layers = [500,250,100,13]
optimal_learning_rate = 0.01
optimal_epoch = 50

# import and extract the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
del df  #delete dataframe to reduce usage of memory
del df_test #delete dataframe to reduce usage of memory

X_train_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_train_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_train_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

X_test_contrast = np.zeros(np.shape(X_test))
for i in range(len(X_test_contrast)):
    image = X_test[i, :]
    image = image.astype(np.uint8)
    X_test_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# normalize data
X_train_contrast = X_train_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test_contrast = X_test_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN


X_train_contrast_noise = np.zeros(np.shape(X_train_contrast))
for i in range(len(X_train_contrast)):
    X_train_contrast_noise[i] = add_noise(X_train_contrast[i, :], noise_type=optimal_noise_type, parameter=optimal_noise)
X_train_contrast_noise = torch.Tensor(X_train_contrast_noise)
X_train_contrast = torch.Tensor(X_train_contrast)
train_ds_clean = TensorDataset(X_train_contrast)
train_ds_noise = TensorDataset(X_train_contrast_noise)
train_dl_clean = DataLoader(train_ds_clean, batch_size=optimal_batch_size, shuffle=False)
train_dl_noise = DataLoader(train_ds_noise, batch_size=optimal_batch_size, shuffle=False)

X_test_contrast = torch.Tensor(X_test_contrast)
test_ds = TensorDataset(X_test_contrast)
visualize_test = DataLoader(test_ds, batch_size=1, shuffle=False)
final_model = Model()
# test_loss, training_loss, autoencoder = final_model.fit(optimal_noise,
#                                                    optimal_batch_size,
#                                                    optimal_hidden_layers,
#                                                    train_dl_clean,
#                                                    train_dl_noise,
#                                                    visualize_test,
#                                                    optimal_noise_type,
#                                                    optimal_epoch,
#                                                    EPOCHS_PRETRAINING,
#                                                    optimal_learning_rate)
#
# pickle.dump(autoencoder,open('final_autoencoder.sav', 'wb'))

autoencoder = pickle.load(open('final_autoencoder.sav', 'rb'))

visualize_train = DataLoader(train_ds_clean, batch_size=1, shuffle=False)
reduced_train = np.zeros((len(visualize_train),13))
for i, features in enumerate(visualize_train):
    reduced_train[i] = autoencoder.encode(features[0]).detach().numpy()
np.savetxt('reduced_trainset_2.csv', reduced_train, delimiter=',')

reduced_test = np.zeros((len(visualize_test),13))
for i, features in enumerate(visualize_test):
    reduced_test[i] = autoencoder.encode(features[0]).detach().numpy()
np.savetxt('reduced_testset_2.csv', reduced_test, delimiter=',')

NUMBER_OF_PICTURES_TO_DISPLAY = 10  # How many pictures we will display
plt.figure(figsize=(20, 4))
for i, features in enumerate(visualize_test):
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

# plt.savefig("final_test_prediction" + " with noise type " + optimal_noise_type +
#           " [" + str(optimal_noise) + "], batch size " + str(optimal_batch_size) +
#           " hidden layers " + ','.join([str(elem) for elem in optimal_hidden_layers]) + " lr " + str(optimal_learning_rate).replace('.', ',') +
#           " epoch " + str(optimal_epoch) )

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Autoencoders.utils import add_noise
from Autoencoders.model import Model
from sklearn.model_selection import KFold
import pickle

autoencoder = pickle.load(open('final_autoencoder.sav', 'rb'))

plt.figure(figsize=(30, 30))
#Extract features
for i in range(25):
    weights = autoencoder.encoders[0].detach().numpy()[:,i+10].reshape(28,28)
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(weights)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()