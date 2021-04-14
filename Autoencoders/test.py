import random

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import cv2
from torch.utils.data import DataLoader, TensorDataset
from Autoencoders.Deep_Autoencoder_model import DenoisingDeepAutoencoder
import pickle


# CONSTANTS
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
EPOCHS_PRETRAINING = 10
EPOCHS_FINETUNING = 10
NUMBER_FOLDS = 5
optimal_batch_size = 64
optimal_pretraining_noise_type = 'gaussian'
optimal_pretraining_noise_parameter = 0.1
optimal_finetuning_noise_type = 'gaussian' # ONLY FOR DENOISING DEEP AUTOENCODER
optimal_finetuning_noise_parameter = 0.3   # ONLY FOR DENOISING DEEP AUTOENCODER
optimal_hidden_layers = [620, 330, 13]
optimal_pretraining_learning_rate = 0.01
optimal_finetuning_learning_rate = 0.001


# import and extract the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")[1500:]
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

model = DenoisingDeepAutoencoder()
test_loss, train_loss, model = model.fit(optimal_pretraining_noise_type,
                                               optimal_pretraining_noise_parameter,
                                               optimal_finetuning_noise_type,   # ONLY FOR DENOISING DEEP AUTOENCODER
                                               optimal_finetuning_noise_parameter,  # ONLY FOR DENOISING DEEP AUTOENCODER
                                               optimal_batch_size,
                                               optimal_hidden_layers,
                                               X_train_contrast,
                                               X_test_contrast,
                                               EPOCHS_FINETUNING,
                                               EPOCHS_PRETRAINING,
                                               optimal_pretraining_learning_rate,
                                               optimal_finetuning_learning_rate)

last_loss = test_loss[-1]
min_loss = test_loss.min()
epoch = test_loss.argmin() + 1

print("last loss:" )
print(last_loss)
print("optimal epoch: ")
print(epoch)
print("optimal loss: ")
print(min_loss)

# Plot the average validation and training losses for this set of hyperparameters
N = np.arange(0, EPOCHS_FINETUNING)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, test_loss, label="test_loss")
plt.plot(N, train_loss, label="final_train_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Mean Squared Error")
plt.legend(loc="lower left")
plt.savefig(f"_EPOCHS_VALIDATION_"
            f"_BATCH_SIZE_{str(optimal_batch_size)}"
            f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
            f"_P_NOISE_PAR_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
            f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}"   # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_F_NOISE_PAR_{str(optimal_finetuning_noise_parameter).replace('.', ',')}" # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
            f"_P_LR_{str(optimal_pretraining_learning_rate).replace('.', ',')}"
            f"F_LR{str(optimal_finetuning_learning_rate).replace('.', ',')}")

# Save the optimal trained model using pickle
pickle.dump(model,open(f"_Autoencoder_"
                       f"_BATCH_SIZE_{str(optimal_batch_size)}"
                       f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
                       f"_P_NOISE_PAR_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
                       f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}"   # ONLY FOR DENOISING DEEP AUTOENCODER
                       f"_F_NOISE_PAR_{str(optimal_finetuning_noise_parameter).replace('.', ',')}" # ONLY FOR DENOISING DEEP AUTOENCODER
                       f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
                       f"_P_LR_{str(optimal_pretraining_learning_rate).replace('.', ',')}"
                       f"_F_LR_{str(optimal_finetuning_learning_rate).replace('.', ',')}"
                       f"_EPOCH_{str(EPOCHS_FINETUNING)}.sav", 'wb'))

# # autoencoder = pickle.load(open('Autoencoder_with_noise__BATCH_SIZE_64_P_NOISE_TYPE_gaussian_P_NOISE_PERCENTAGE_0,1_F_NOISE_TYPE_gaussian_F_NOISE_PERCENTAGE_0,3_LAYERS_[620,330,13]_LR_0,001_EPOCH_27.sav', 'rb'))

# Save the lower dimensional training dataset as predicted by the optimal autoencoder
X_train_torch = torch.Tensor(X_train_contrast)
train_ds = TensorDataset(X_train_torch)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
reduced_train = np.zeros((len(train_dl),13))
for i, features in enumerate(train_dl):
    reduced_train[i] = model.encode(features[0]).detach().numpy()
np.savetxt(f"reduced_trainset_"
           f"_BATCH_SIZE_{str(optimal_batch_size)}"
           f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
           f"_P_NOISE_PAR_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
           f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}" # ONLY FOR DENOISING DEEP AUTOENCODER
           f"_F_NOISE_PAR_{str(optimal_finetuning_noise_parameter).replace('.', ',')}"   # ONLY FOR DENOISING DEEP AUTOENCODER
           f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
           f"_P_LR_{str(optimal_pretraining_learning_rate).replace('.', ',')}"
           f"_F_LR_{str(optimal_finetuning_learning_rate).replace('.', ',')}"
           f"_EPOCH_{str(EPOCHS_FINETUNING)}.csv", reduced_train, delimiter=',')

# Save the lower dimensional test dataset as predicted by the optimal autoencoder
X_test_contrast = torch.Tensor(X_test_contrast)
test_ds = TensorDataset(X_test_contrast)
visualize_test = DataLoader(test_ds, batch_size=1, shuffle=False)
reduced_test = np.zeros((len(visualize_test),13))
for i, features in enumerate(visualize_test):
    reduced_test[i] = model.encode(features[0]).detach().numpy()
np.savetxt(f"reduced_testset_"
           f"_BATCH_SIZE_{str(optimal_batch_size)}"
           f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
           f"_P_NOISE_PAR_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
           f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}" # ONLY FOR DENOISING DEEP AUTOENCODER
           f"_F_NOISE_PAR_{str(optimal_finetuning_noise_parameter).replace('.', ',')}"   # ONLY FOR DENOISING DEEP AUTOENCODER
           f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
           f"_P_LR_{str(optimal_pretraining_learning_rate).replace('.', ',')}"
           f"_F_LR_{str(optimal_finetuning_learning_rate).replace('.', ',')}"
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
    plt.imshow(model(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 9:
        break

plt.savefig(f"_test_predictions_"
            f"_BATCH_SIZE_{str(optimal_batch_size)}"
            f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
            f"_P_NOISE_PAR_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
            f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}" # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_F_NOISE_PAR_{str(optimal_finetuning_noise_parameter).replace('.', ',')}"   # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
            f"_P_LR_{str(optimal_pretraining_learning_rate).replace('.', ',')}"
            f"_F_LR_{str(optimal_finetuning_learning_rate).replace('.', ',')}"
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
    plt.imshow(model(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 9:
        break

plt.savefig(f"_train_predictions_"
            f"_BATCH_SIZE_{str(optimal_batch_size)}"
            f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
            f"_P_NOISE_PAR_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
            f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}" # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_F_NOISE_PAR_{str(optimal_finetuning_noise_parameter).replace('.', ',')}"   # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
            f"_P_LR_{str(optimal_pretraining_learning_rate).replace('.', ',')}"
            f"_F_LR_{str(optimal_finetuning_learning_rate).replace('.', ',')}"
            f"_EPOCHS_{str(EPOCHS_FINETUNING)}")

# Analyse the features that are captured by the first layer of the model
plt.figure(figsize=(50, 50))
for i in range(225):
    weights = model.encoders[0].detach().numpy()[:,i+1].reshape(28,28)
    ax = plt.subplot(15, 15, i + 1)
    plt.imshow(weights)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(f"features_captured_with_noise"
            f"_BATCH_SIZE_{str(optimal_batch_size)}"
            f"_P_NOISE_TYPE_{optimal_pretraining_noise_type}"
            f"_P_NOISE_PAR_{str(optimal_pretraining_noise_parameter).replace('.', ',')}"
            f"_F_NOISE_TYPE_{optimal_finetuning_noise_type}" # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_F_NOISE_PAR_{str(optimal_finetuning_noise_parameter).replace('.', ',')}"   # ONLY FOR DENOISING DEEP AUTOENCODER
            f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
            f"_P_LR_{str(optimal_pretraining_learning_rate).replace('.', ',')}"
            f"_F_LR_{str(optimal_finetuning_learning_rate).replace('.', ',')}"
            f"_EPOCH_{str(EPOCHS_FINETUNING)}")

