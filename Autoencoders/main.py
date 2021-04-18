import random
import pickle
import matplotlib
import torch
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from Autoencoders.Deep_Autoencoder_model import DenoisingDeepAutoencoder
from Autoencoders.Deep_Autoencoder_model import DeepAutoencoder
from sklearn.model_selection import KFold
from Autoencoders.utils import create_title, reconstruct
from constants import MAX_BRIGHTNESS, INTRINSIC_DIMENSIONALITY, PICTURE_DIMENSION, NUMBER_OF_PIXELS, MEAN
matplotlib.use('TkAgg')

# CONSTANTS
EPOCHS_PRETRAINING = 20
EPOCHS_FINETUNING = 50
NUMBER_FOLDS = 5
NUMBER_COMBINATIONS = 10

# Hyper-parameters to be tuned using cross-validation
BATCH_SIZE = [8, 16, 32, 64, 128]
NOISE_PRETRAINING = {'zeros': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'gaussian': [0, 0.3, 0.5, 0.75, 1, 1.75, 2]}
NOISE_FINETUNING = {'zeros': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'gaussian': [0, 0.3, 0.5, 0.75, 1, 1.75, 2]}
HIDDEN_LAYERS = [[800, 400, 200, INTRINSIC_DIMENSIONALITY], [800, 250, INTRINSIC_DIMENSIONALITY],
                 [620, 330, 100, INTRINSIC_DIMENSIONALITY], [500, 250, 100, INTRINSIC_DIMENSIONALITY],
                 [500, 250, INTRINSIC_DIMENSIONALITY], [620, 330, INTRINSIC_DIMENSIONALITY]]
LEARNING_RATE_PRETRAINING = [0.001, 0.002, 0.003, 0.005, 0.01]
LEARNING_RATE_FINETUNING = [0.001, 0.002, 0.003, 0.005, 0.01]

# Import and extract the training data
df = pd.read_csv("../Data/sign_mnist_train.csv")

# Extract the training features and labels into numpy
X_train = df.iloc[:, 1:].values
y_train = df.iloc[:, 0].values

# Delete dataframes to reduce usage of memory
del df

# Create the contrasted training set
X_train_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_train_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_train_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# Normalize normal and contrasted training sets
X_train_contrast = X_train_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN

# Extract the folds for cross-validation
kf = KFold(n_splits=NUMBER_FOLDS)

# Initialise the optimal parameters that will be used to train the optimal model
optimal_loss = 0
optimal_noise = None
optimal_pretraining_noise_parameter = 0
optimal_pretraining_noise_type = None
optimal_finetuning_noise_parameter = 0
optimal_finetuning_noise_type = None
optimal_batch_size = 0
optimal_hidden_layers = None
optimal_epoch = 0
optimal_pretraining_learning_rate = 0
optimal_finetuning_learning_rate = 0

# Cross-validation
for i in range(NUMBER_COMBINATIONS):
    # Randomly select one combination of the hyperparameters
    pretraining_noise_type = random.sample(NOISE_PRETRAINING.keys(), 1)[0]
    pretraining_noise_parameter = random.sample(NOISE_PRETRAINING[pretraining_noise_type], 1)[0]
    finetuning_noise_type = random.sample(NOISE_FINETUNING.keys(), 1)[0]
    finetuning_noise_parameter = random.sample(NOISE_FINETUNING[finetuning_noise_type], 1)[0]
    batch_size = random.sample(BATCH_SIZE, 1)[0]
    hidden_layers = random.sample(HIDDEN_LAYERS, 1)[0]
    learning_rate_pretraining = random.sample(LEARNING_RATE_PRETRAINING, 1)[0]
    learning_rate_finetuning = random.sample(LEARNING_RATE_FINETUNING, 1)[0]

    # Print the Cross-validation that is being run to keep track of the process
    print(f"Starting CV {str(i + 1)} of " +
          create_title(batch_size, pretraining_noise_type, pretraining_noise_parameter, hidden_layers,
                       learning_rate_pretraining, learning_rate_finetuning,
                       finetuning_noise_type, finetuning_noise_parameter))  # ONLY FOR DENOSING AUTOENCODERS

    # Initialize the matrices that will save the training and validation losses for the different folds
    current_validation_losses = np.zeros((EPOCHS_FINETUNING, NUMBER_FOLDS))
    current_final_training_losses = np.zeros((EPOCHS_FINETUNING, NUMBER_FOLDS))
    column = 0

    # Loop over all the folds
    for train_index, test_index in kf.split(X_train_contrast):
        print(f"Starting_Fold_{str(column + 1)}")

        # Create the training and validation sets for this fold
        X_train_CV, X_validation_CV = X_train_contrast[train_index], X_train_contrast[test_index]

        # Initialize and fit the model
        model = DenoisingDeepAutoencoder()
        val_loss, final_train, model = model.fit(pretraining_noise_type,
                                                 pretraining_noise_parameter,
                                                 finetuning_noise_type,  # ONLY FOR DENOISING DEEP AUTOENCODER
                                                 finetuning_noise_parameter,  # ONLY FOR DENOISING DEEP AUTOENCODER
                                                 batch_size,
                                                 hidden_layers,
                                                 X_train_CV,
                                                 X_validation_CV,
                                                 EPOCHS_FINETUNING,
                                                 EPOCHS_PRETRAINING,
                                                 learning_rate_pretraining,
                                                 learning_rate_finetuning)

        # Save the validation loss and training loss of this fold
        current_validation_losses[:, column] = val_loss
        current_final_training_losses[:, column] = final_train
        column += 1

    # Compute the average validation loss and train loss to be able to plot it
    average_validation_loss = current_validation_losses.mean(axis=1)
    average_final = current_final_training_losses.mean(axis=1)
    print(average_validation_loss[-1])
    print(average_final[-1])

    # Find the minimum val loss to choose the optimal model
    minimum_loss = average_validation_loss.min()
    epoch = average_validation_loss.argmin() + 1

    # Plot the average validation and training losses for this set of hyperparameters
    N = np.arange(0, EPOCHS_FINETUNING)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, average_validation_loss, label="val_loss")
    plt.plot(N, average_final, label="final_train_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(f"loss_graph_" + create_title(batch_size, pretraining_noise_type, pretraining_noise_parameter,
                                              hidden_layers, learning_rate_pretraining, learning_rate_finetuning,
                                              EPOCHS_PRETRAINING, EPOCHS_FINETUNING,
                                              finetuning_noise_type,
                                              finetuning_noise_parameter))  # ONLY FOR DENOSING AUTOENCODERS

    # Save the optimal hyperparameter set
    if i == 0 or minimum_loss < optimal_loss:
        optimal_loss = minimum_loss
        optimal_pretraining_noise_parameter = pretraining_noise_parameter
        optimal_pretraining_noise_type = pretraining_noise_type
        optimal_finetuning_noise_parameter = finetuning_noise_parameter
        optimal_finetuning_noise_type = finetuning_noise_type
        optimal_batch_size = batch_size
        optimal_hidden_layers = hidden_layers
        optimal_epoch = epoch
        optimal_finetuning_learning_rate = learning_rate_finetuning
        optimal_pretraining_learning_rate = learning_rate_pretraining

print("Optimal model is" +
      create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                   optimal_hidden_layers, optimal_pretraining_learning_rate, optimal_finetuning_learning_rate,
                   EPOCHS_PRETRAINING, optimal_epoch,
                   optimal_finetuning_noise_type, optimal_finetuning_noise_parameter,
                   optimal_loss))  # ONLY FOR DENOSING AUTOENCODERS

print("Starting the validation of the number of epochs")

# Import and extract the training data
df_test = pd.read_csv("../Data/sign_mnist_test.csv")

# Use the first 1500 observations of the test set to cross-validate the optimal number of epochs
df_validation = df_test[0:1500]
df_test = df_test[1500:]

# Extract the validation features and labels into numpy
X_validation = df_validation.iloc[:, 1:].values
y_validation = df_validation.iloc[:, 0].values

# Extract the test features and labels into numpy
X_test = df_validation.iloc[:, 1:].values
y_test = df_validation.iloc[:, 0].values

# Delete dataframes to reduce usage of memory
del df_test

# Create the contrasted test set
X_validation_contrast = np.zeros(np.shape(X_validation))
for i in range(len(X_validation_contrast)):
    image = X_validation[i, :]
    image = image.astype(np.uint8)
    X_validation_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# Create the contrasted test set
X_test_contrast = np.zeros(np.shape(X_test))
for i in range(len(X_test_contrast)):
    image = X_test[i, :]
    image = image.astype(np.uint8)
    X_test_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# Normalize normal and contrasted test sets
X_validation_contrast = X_test_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_validation = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test_contrast = X_test_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

# Initialize the model and train it with the full training set and with the optimal hyperparameters
final_model = DenoisingDeepAutoencoder()
test_loss, training_loss, final_model = final_model.fit(optimal_pretraining_noise_type,
                                                        optimal_pretraining_noise_parameter,
                                                        optimal_finetuning_noise_type,
                                                        # ONLY FOR DENOISING DEEP AUTOENCODER
                                                        optimal_finetuning_noise_parameter,
                                                        # ONLY FOR DENOISING DEEP AUTOENCODER
                                                        optimal_batch_size,
                                                        optimal_hidden_layers,
                                                        X_train_contrast,
                                                        X_validation_contrast,
                                                        optimal_epoch,
                                                        EPOCHS_PRETRAINING,
                                                        optimal_pretraining_learning_rate,
                                                        optimal_finetuning_learning_rate)

optimal_epoch = np.array(test_loss).argmin() + 1
print("The optimal number of epochs is: ", optimal_epoch)
print(test_loss)
print(training_loss)

# Plot the average validation and training losses for this set of hyperparameters
N = np.arange(0, EPOCHS_FINETUNING)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, test_loss, label="test_loss")
plt.plot(N, training_loss, label="final_train_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("_EPOCHS_VALIDATION_" +
            create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                         optimal_hidden_layers, optimal_pretraining_learning_rate, optimal_finetuning_learning_rate,
                         EPOCHS_PRETRAINING, optimal_epoch,
                         optimal_finetuning_noise_type,
                         optimal_finetuning_noise_parameter))  # ONLY FOR DENOSING AUTOENCODERS

# Initialize the model and train it with the full training set and with the optimal hyperparameters
final_model_early_stopping = DenoisingDeepAutoencoder()
test_loss, training_loss, final_model_early_stopping = final_model_early_stopping.fit(optimal_pretraining_noise_type,
                                                                                      optimal_pretraining_noise_parameter,
                                                                                      optimal_finetuning_noise_type,
                                                                                      # ONLY FOR DENOISING DEEP AUTOENCODER
                                                                                      optimal_finetuning_noise_parameter,
                                                                                      # ONLY FOR DENOISING DEEP AUTOENCODER
                                                                                      optimal_batch_size,
                                                                                      optimal_hidden_layers,
                                                                                      X_train_contrast,
                                                                                      X_test_contrast,
                                                                                      optimal_epoch,
                                                                                      EPOCHS_PRETRAINING,
                                                                                      optimal_pretraining_learning_rate,
                                                                                      optimal_finetuning_learning_rate)

# Save the optimal trained model using pickle
pickle.dump(final_model_early_stopping,
            open("_Autoencoder_" +
                 create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                              optimal_hidden_layers, optimal_pretraining_learning_rate,
                              optimal_finetuning_learning_rate,
                              EPOCHS_PRETRAINING, optimal_epoch,
                              optimal_finetuning_noise_type,
                              optimal_finetuning_noise_parameter) +  # ONLY FOR DENOSING AUTOENCODERS
                 ".sav", 'wb'))

# If we need to retrive the model we can use this
# autoencoder = pickle.load(open('final_autoencoder.sav', 'rb'))

# Save the lower dimensional training dataset as predicted by the optimal autoencoder
X_train_torch = torch.Tensor(X_train_contrast)
train_ds = TensorDataset(X_train_torch)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
reduced_train = np.zeros((len(train_dl), INTRINSIC_DIMENSIONALITY))
for i, features in enumerate(train_dl):
    reduced_train[i] = final_model_early_stopping.encode(features[0]).detach().numpy()
np.savetxt("reduced_trainset_" +
           create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                        optimal_hidden_layers, optimal_pretraining_learning_rate, optimal_finetuning_learning_rate,
                        EPOCHS_PRETRAINING, optimal_epoch,
                        optimal_finetuning_noise_type,
                        optimal_finetuning_noise_parameter) +  # ONLY FOR DENOSING AUTOENCODERS
           ".csv", reduced_train, delimiter=',')

# Save the lower dimensional test dataset as predicted by the optimal autoencoder
X_test_contrast = torch.Tensor(X_test_contrast)
test_ds = TensorDataset(X_test_contrast)
visualize_test = DataLoader(test_ds, batch_size=1, shuffle=False)
reduced_test = np.zeros((len(visualize_test), INTRINSIC_DIMENSIONALITY))
for i, features in enumerate(visualize_test):
    reduced_test[i] = final_model_early_stopping.encode(features[0]).detach().numpy()
np.savetxt("reduced_testset_" +
           create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                        optimal_hidden_layers, optimal_pretraining_learning_rate, optimal_finetuning_learning_rate,
                        EPOCHS_PRETRAINING, optimal_epoch,
                        optimal_finetuning_noise_type,
                        optimal_finetuning_noise_parameter) +  # ONLY FOR DENOSING AUTOENCODERS
           ".csv", reduced_test, delimiter=',')

reconstruct(visualize_test, final_model_early_stopping, optimal_batch_size, optimal_pretraining_noise_type,
            optimal_pretraining_noise_parameter, optimal_hidden_layers, optimal_pretraining_learning_rate,
            optimal_finetuning_learning_rate, EPOCHS_PRETRAINING, optimal_epoch, "Test_predictions_",
            optimal_finetuning_noise_type, optimal_finetuning_noise_parameter)  # ONLY FOR DENOSING AUTOENCODERS

reconstruct(train_dl, final_model_early_stopping, optimal_batch_size, optimal_pretraining_noise_type,
            optimal_pretraining_noise_parameter, optimal_hidden_layers, optimal_pretraining_learning_rate,
            optimal_finetuning_learning_rate, EPOCHS_PRETRAINING, optimal_epoch, "Train_predictions_",
            optimal_finetuning_noise_type, optimal_finetuning_noise_parameter)  # ONLY FOR DENOSING AUTOENCODERS

# Analyse the features that are captured by the first layer of the model
plt.figure(figsize=(30, 30))
for i in range(150):
    weights = final_model_early_stopping.encoders[0].detach().numpy()[:, i + 3].reshape(PICTURE_DIMENSION,
                                                                                        PICTURE_DIMENSION)
    ax = plt.subplot(15, 10, i + 1)
    plt.imshow(weights)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(f"features_captured_with_noise" +
            create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                         optimal_hidden_layers, optimal_pretraining_learning_rate, optimal_finetuning_learning_rate,
                         EPOCHS_PRETRAINING, optimal_epoch,
                         optimal_finetuning_noise_type,
                         optimal_finetuning_noise_parameter))  # ONLY FOR DENOSING AUTOENCODERS
