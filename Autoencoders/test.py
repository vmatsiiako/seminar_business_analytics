import random

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Autoencoders.model import Model
import pickle


# CONSTANTS
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
EPOCHS_PRETRAINING = 5
EPOCHS_FINETUNING = 10
NUMBER_FOLDS = 5
optimal_batch_size = 32
optimal_pretraining_noise_type = 'gaussian'
optimal_pretraining_noise_parameter = 2
optimal_finetuning_noise_type = 'zeros' # ONLY FOR DENOISING DEEP AUTOENCODER
optimal_finetuning_noise_parameter = 0.2    # ONLY FOR DENOISING DEEP AUTOENCODER
optimal_hidden_layers = [620, 330, 13]  #[800,400,200,13],[800, 250, 13],[620, 330, 13]
optimal_learning_rate = 0.002  # 0.001

# best model: 32, gaussian 2, zeros 0.4, [620, 330, 13], 0.002 epochs 20 20
# best model: 32, gaussian 2, zeros 0.2, [620, 330, 13], 0.002 wpochs 20 20


# import and extract the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_train = df.iloc[:,1:].values
y_train = df.iloc[:,0].values
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,0].values
del df  #delete dataframe to reduce usage of memory
del df_test #delete dataframe to reduce usage of memory

# Construct the contrasted training pictures dataset
X_train_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_train_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_train_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# Construct the contrasted test pictures dataset
X_test_contrast = np.zeros(np.shape(X_test))
for i in range(len(X_test_contrast)):
    image = X_test[i, :]
    image = image.astype(np.uint8)
    X_test_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# Normalize data
X_train_contrast = X_train_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test_contrast = X_test_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

model = Model()
test_loss, train_loss, autoencoder = model.fit(optimal_pretraining_noise_type,
                                               optimal_pretraining_noise_parameter,
                                               optimal_finetuning_noise_type,   # ONLY FOR DENOISING DEEP AUTOENCODER
                                               optimal_finetuning_noise_parameter,  # ONLY FOR DENOISING DEEP AUTOENCODER
                                               optimal_batch_size,
                                               optimal_hidden_layers,
                                               X_train_contrast,
                                               X_test_contrast,
                                               EPOCHS_FINETUNING,
                                               EPOCHS_PRETRAINING,
                                               optimal_learning_rate)

print(test_loss)

# Save the optimal trained model using pickle
pickle.dump(autoencoder,open(f"Autoencoder_with_noise_"
                             f"_BATCH_SIZE_{str(optimal_batch_size)}"
                             f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
                             f"_P_NOISE_PERCENTAGE_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
                             f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}"   # ONLY FOR DENOISING DEEP AUTOENCODER
                             f"_F_NOISE_PERCENTAGE_{str(optimal_finetuning_noise_parameter).replace('.', ',')}" # ONLY FOR DENOISING DEEP AUTOENCODER
                             f"_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
                             f"_LR_{str(optimal_learning_rate).replace('.', ',')}"
                             f"_EPOCH_{str(EPOCHS_FINETUNING)}.sav", 'wb'))

# autoencoder = pickle.load(open('Autoencoder_with_noise__BATCH_SIZE_32_P_NOISE_TYPE_gaussian_P_NOISE_PERCENTAGE_1,25_F_NOISE_TYPE_zeros_F_NOISE_PERCENTAGE_0,5_LAYERS_[800,250,13]_LR_0,003_EPOCH_15.sav', 'rb'))

# Save the lower dimensional training dataset as predicted by the optimal autoencoder
X_train_torch = torch.Tensor(X_train_contrast)
train_ds = TensorDataset(X_train_torch)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
reduced_train = np.zeros((len(train_dl),13))
for i, features in enumerate(train_dl):
    reduced_train[i] = autoencoder.encode(features[0]).detach().numpy()
np.savetxt(f"reduced_trainset_with_noise_"
           f"_BATCH_SIZE_{str(optimal_batch_size)}"
           f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
           f"_P_NOISE_PERCENTAGE_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
           f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}" # ONLY FOR DENOISING DEEP AUTOENCODER
           f"_F_NOISE_PERCENTAGE_{str(optimal_finetuning_noise_parameter).replace('.', ',')}"   # ONLY FOR DENOISING DEEP AUTOENCODER
           f"_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
           f"_LR_{str(optimal_learning_rate).replace('.', ',')}"
           f"_EPOCH_{str(EPOCHS_FINETUNING)}.csv", reduced_train, delimiter=',')

# Save the lower dimensional test dataset as predicted by the optimal autoencoder
X_test_contrast = torch.Tensor(X_test_contrast)
test_ds = TensorDataset(X_test_contrast)
visualize_test = DataLoader(test_ds, batch_size=1, shuffle=False)
reduced_test = np.zeros((len(visualize_test),13))
for i, features in enumerate(visualize_test):
    reduced_test[i] = autoencoder.encode(features[0]).detach().numpy()
np.savetxt(f"reduced_test_set_with_noise"
      f"_BATCH_SIZE_{str(optimal_batch_size)}"
      f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
      f"_P_NOISE_PERCENTAGE_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
      f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}"  # ONLY FOR DENOISING DEEP AUTOENCODER
      f"_F_NOISE_PERCENTAGE_{str(optimal_finetuning_noise_parameter).replace('.', ',')}"    # ONLY FOR DENOISING DEEP AUTOENCODER
      f"_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
      f"_LR_{str(optimal_learning_rate).replace('.', ',')}"
      f"_EPOCH_{str(EPOCHS_FINETUNING)}.csv", reduced_test, delimiter=',')

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

plt.savefig(f"finale_test_predictions_with_noise"
            f"_BS_{str(optimal_batch_size)}"
            f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
            f"_P_NOISE_PERE_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
            f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}"    # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_F_NOISE_PER_{str(optimal_finetuning_noise_parameter).replace('.', ',')}" # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
            f"_LR_{str(optimal_learning_rate).replace('.', ',')}"
            f"_EPOCHS_{str(EPOCHS_FINETUNING)}")

NUMBER_OF_PICTURES_TO_DISPLAY = 10  # How many pictures we will display
plt.figure(figsize=(20, 4))
for i, features in enumerate(train_dl):
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

plt.savefig(f"finale_train_predictions_with_noise"
            f"_BS_{str(optimal_batch_size)}"
            f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
            f"_P_NOISE_PER_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
            f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}"    # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_F_NOISE_PER_{str(optimal_finetuning_noise_parameter).replace('.', ',')}" # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
            f"_LR_{str(optimal_learning_rate).replace('.', ',')}"
            f"_EPOCHS_{str(EPOCHS_FINETUNING)}")

# Analyse the features that are captured by the first layer of the model
plt.figure(figsize=(30, 30))
for i in range(150):
    weights = autoencoder.encoders[0].detach().numpy()[:,i+3].reshape(28,28)
    ax = plt.subplot(15, 10, i + 1)
    plt.imshow(weights)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(f"features_captured_with_noise"
            f"_BS_{str(optimal_batch_size)}"
            f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
            f"_P_NOISE_PER_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
            f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}"    # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_F_NOISE_PER_{str(optimal_finetuning_noise_parameter).replace('.', ',')}" # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
            f"_LR_{str(optimal_learning_rate).replace('.', ',')}"
            f"_EPOCH_{str(EPOCHS_FINETUNING)}")

# visualize_train = DataLoader(train_ds_clean, batch_size=1, shuffle=False)
# reduced_train = np.zeros((len(visualize_train),13))
# for i, features in enumerate(visualize_train):
#     reduced_train[i] = autoencoder.encode(features[0]).detach().numpy()
# np.savetxt(f"reduced_trainset_with"
#       f"_BATCH_SIZE_{str(optimal_batch_size)}"
#       f"_NOISE_TYPE_{optimal_noise_type}"
#       f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
#       f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
#       f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
#       f"_EPOCH_{str(optimal_epoch)}.csv", reduced_train, delimiter=',')
#
# reduced_test = np.zeros((len(visualize_test),13))
# for i, features in enumerate(visualize_test):
#     reduced_test[i] = autoencoder.encode(features[0]).detach().numpy()
# np.savetxt(f"reduced_test_set_with"
#       f"_BATCH_SIZE_{str(optimal_batch_size)}"
#       f"_NOISE_TYPE_{optimal_noise_type}"
#       f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
#       f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
#       f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
#       f"_EPOCH_{str(optimal_epoch)}.csv", reduced_test, delimiter=',')
#
# NUMBER_OF_PICTURES_TO_DISPLAY = 10  # How many pictures we will display
# plt.figure(figsize=(20, 4))
# for i, features in enumerate(visualize_test):
#     # Display original
#     ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1)
#     plt.imshow(features[0].numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # Display reconstruction
#     ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1 + NUMBER_OF_PICTURES_TO_DISPLAY)
#     plt.imshow(autoencoder(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     if i == 9:
#         break
#
# plt.savefig(f"finale_test_predictions_with_noise"
#             f"_BATCH_SIZE_{str(optimal_batch_size)}"
#             f"_NOISE_TYPE_{optimal_noise_type}"
#             f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
#             f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
#             f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
#             f"_EPOCHS_{str(optimal_epoch)}")
#
# NUMBER_OF_PICTURES_TO_DISPLAY = 10  # How many pictures we will display
# plt.figure(figsize=(20, 4))
# for i, features in enumerate(visualize_train):
#     # Display original
#     ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1)
#     plt.imshow(features[0].numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # Display reconstruction
#     ax = plt.subplot(2, NUMBER_OF_PICTURES_TO_DISPLAY, i + 1 + NUMBER_OF_PICTURES_TO_DISPLAY)
#     plt.imshow(autoencoder(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     if i == 9:
#         break
#
# plt.savefig(f"finale_train_predictions_with_noise"
#             f"_BATCH_SIZE_{str(optimal_batch_size)}"
#             f"_NOISE_TYPE_{optimal_noise_type}"
#             f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
#             f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
#             f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
#             f"_EPOCHS_{str(optimal_epoch)}")
#
# # Analyse the features that are captured by the first layer of the model
# plt.figure(figsize=(30, 30))
# for i in range(150):
#     weights = autoencoder.encoders[0].detach().numpy()[:,i+3].reshape(28,28)
#     ax = plt.subplot(15, 10, i + 1)
#     plt.imshow(weights)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
# plt.savefig(f"features_captured_with_noise"
#       f"_BATCH_SIZE_{str(optimal_batch_size)}"
#       f"_NOISE_TYPE_{optimal_noise_type}"
#       f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
#       f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
#       f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
#       f"_EPOCH_{str(optimal_epoch)}")