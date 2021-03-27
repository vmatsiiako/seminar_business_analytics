import random
import pickle
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


# CONSTANTS
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
BATCH_SIZE = [32, 64]
NOISE = {'zeros': [0, 0.1, 0.2, 0.3, 0.5, 0.6], 'gaussian': [1, 1.25, 1.5, 1.75, 2, 2.25]}
HIDDEN_LAYERS = [[500, 250, 100, 13], [500, 250, 13], [1000, 500, 250, 13]]
LEARNING_RATE = [0.005, 0.0075, 0.01, 0.0125, 0.015]
EPOCHS_PRETRAINING = 20
EPOCHS_FINETUNING = 60
NUMBER_FOLDS = 5

#Models to try
#original: bs-64, gaussian, noise = 1, layers = [500,250,100,13], lr = 0.01, epochs fine= 50

# bs-32, gaussian, noise = 1, layers = [500,250,100,13], lr = 0.01, epochs fine= 70
# bs-32, gaussian, noise = 2, layers = [500,250,100,13], lr = 0.01, epochs fine= 70
# bs-64, gaussian, noise = 2, layers = [500,250,100,13], lr = 0.01, epochs fine= 70
# bs-64, gaussian, noise = 1, layers = [500,250,100,13], lr = 0.005, epochs fine= 70
# bs-32, gaussian, noise = 1, layers = [500,250,100,13], lr = 0.005, epochs fine= 70



# import and extract the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")

#Extract the training features and labels into numpy
X_train = df.iloc[:,1:].values 
y_train = df.iloc[:,0].values

#Extract the test features and labels into numpy
X_test = df_test.iloc[:,1:].values 
y_test = df_test.iloc[:,0].values

#delete dataframes to reduce usage of memory
del df
del df_test

#Create the contrasted training set
X_train_contrast = np.zeros(np.shape(X_train))
for i in range(len(X_train_contrast)):
    image = X_train[i, :]
    image = image.astype(np.uint8)
    X_train_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

#Create the contraste test set
X_test_contrast = np.zeros(np.shape(X_test))
for i in range(len(X_test_contrast)):
    image = X_test[i, :]
    image = image.astype(np.uint8)
    X_test_contrast[i] = cv2.equalizeHist(image).reshape(1, NUMBER_OF_PIXELS)

# normalize normal and contrasted training sets
X_train_contrast = X_train_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_train = X_train.astype('float32') / MAX_BRIGHTNESS - MEAN

# normalize normal and contrasted test sets
X_test_contrast = X_test_contrast.astype('float32') / MAX_BRIGHTNESS - MEAN
X_test = X_test.astype('float32') / MAX_BRIGHTNESS - MEAN

#extract the folds for cross-validation
kf = KFold(n_splits=NUMBER_FOLDS)

#initialise the optimal parameters that will be used to train the optimal model
optimal_loss = 0
optimal_noise = None
optimal_noise_type = None
optimal_batch_size = None
optimal_hidden_layers = None
optimal_epoch = None
optimal_learning_rate = None

#cross-validation
for i in range(12):
    noise_type = random.sample(NOISE.keys(), 1)[0]
    noise_parameter = random.sample(NOISE[noise_type], 1)[0]
    batch_size = random.sample(BATCH_SIZE, 1)[0]
    hidden_layers = random.sample(HIDDEN_LAYERS, 1)[0]
    learning_rate = random.sample(LEARNING_RATE, 1)[0]
    print(f"Starting_CV_{str(i+1)}"
          f"_BATCH_SIZE_{str(batch_size)}"
          f"_NOISE_TYPE_{noise_type}"
          f"_NOISE_PERCENTAGE_{str(noise_parameter).replace('.', ',')}"
          f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in hidden_layers])}]"
          f"_LEATNING_RATE_{str(learning_rate).replace('.', ',')}")

    current_validation_losses = np.zeros((EPOCHS_FINETUNING,NUMBER_FOLDS))
    current_final_training_losses = np.zeros((EPOCHS_FINETUNING, NUMBER_FOLDS))
    column = 0

    for train_index, test_index in kf.split(X_train_contrast):
        print(f"Starting_Fold_{str(column+1)}")
        X_train_CV, X_validation_CV = X_train_contrast[train_index], X_train_contrast[test_index]

        # Convert the data to torch types
        X_train_clean = torch.Tensor(X_train_CV)
        X_validation_clean = torch.Tensor(X_validation_CV)

        # construct the noised train set
        X_train_noise = np.zeros(np.shape(X_train_CV))
        for i in range(len(X_train_CV)):
            X_train_noise[i] = add_noise(X_train_CV[i, :], noise_type=noise_type, parameter=noise_parameter)
        X_train_noise = torch.Tensor(X_train_noise)

        # Create the TensorDataset and the DataLoader
        train_ds_clean = TensorDataset(X_train_clean)
        train_ds_noise = TensorDataset(X_train_noise)
        validation_ds = TensorDataset(X_validation_clean)
        train_dl_clean = DataLoader(train_ds_clean, batch_size=batch_size, shuffle=False)
        train_dl_noise = DataLoader(train_ds_noise, batch_size=batch_size, shuffle=False)
        validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

        # train_dl_clean = DataLoader(train_ds_clean, batch_size=batch_size, shuffle=True)
        # train_dl_noise = DataLoader(train_ds_noise, batch_size=batch_size, shuffle=True)
        # validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)

        #Initialise and fit the model
        model = Model()
        val_loss, final_train, ae = model.fit(noise_parameter,
                                              batch_size,
                                              hidden_layers,
                                              train_dl_clean,
                                              train_dl_noise,
                                              validation_dl,
                                              noise_type,
                                              EPOCHS_FINETUNING,
                                              EPOCHS_PRETRAINING,
                                              learning_rate)

        # Save the validation loss and training loss of this fold
        current_validation_losses[:, column] = val_loss
        current_final_training_losses[:, column] = final_train
        column += 1

    # Compute the average validation loss and train loss to be able to plot it
    average_validation_loss = current_validation_losses.mean(axis=1)
    average_final = current_final_training_losses.mean(axis=1)

    # Find the minimum val loss to choose the optimal model
    minimum_loss = average_validation_loss.min()
    epoch = average_validation_loss.argmin()+1

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
    plt.savefig(f"loss_graph"
                f"_BATCH_SIZE_{str(batch_size)}"
                f"_NOISE_TYPE_{noise_type}"
                f"_NOISE_PERCENTAGE_{str(noise_parameter).replace('.', ',')}"
                f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in hidden_layers])}]"
                f"_LEATNING_RATE_{str(learning_rate).replace('.', ',')}")

    # Save the optimal hyperparameter set
    if i == 0 or minimum_loss < optimal_loss:
        optimal_loss = minimum_loss
        optimal_noise = noise_parameter
        optimal_noise_type = noise_type
        optimal_batch_size = batch_size
        optimal_hidden_layers = hidden_layers
        optimal_epoch = epoch
        optimal_learning_rate = learning_rate

print(f"Optimal_model_is"
      f"_BATCH_SIZE_{str(optimal_batch_size)}"
      f"_NOISE_TYPE_{optimal_noise_type}"
      f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
      f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
      f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
      f"_EPOCH_{str(optimal_epoch)}"
      f"_LOSS_{str(optimal_loss).replace('.', ',')}")

# print("The optimal model is " + " with noise type " + optimal_noise_type +
#       " [" + str(optimal_noise) + "], batch size " + str(optimal_batch_size) +
#       " hidden layers " + ','.join([str(elem) for elem in optimal_hidden_layers]) + " lr " + str(optimal_learning_rate) +
#       " epoch " + str(optimal_epoch) + " loss " + str(optimal_loss))

# X_train_noise = np.zeros(np.shape(X_train_CV))
# for i in range(len(X_train_CV)):
#     X_train_noise[i] = add_noise(X_train_CV[i, :], noise_type=noise_type, parameter=noise_parameter)
# X_train_noise = torch.Tensor(X_train_noise)

# train_ds_clean = TensorDataset(X_train_clean)
# train_ds_noise = TensorDataset(X_train_noise)
# validation_ds = TensorDataset(X_validation_clean)

# Create noised dataset for pretraining on the full training set with optimal hyperparameters
X_train_contrast_noise = np.zeros(np.shape(X_train_contrast))
for i in range(len(X_train_contrast)):
    X_train_contrast_noise[i] = add_noise(X_train_contrast[i, :], noise_type=optimal_noise_type, parameter=optimal_noise)

# Convert numpy to tensors
X_train_contrast_noise = torch.Tensor(X_train_contrast_noise)
X_train_contrast = torch.Tensor(X_train_contrast)
X_test_contrast = torch.Tensor(X_test_contrast)

# Create TensorDataset
train_ds_clean = TensorDataset(X_train_contrast)
train_ds_noise = TensorDataset(X_train_contrast_noise)
test_ds = TensorDataset(X_test_contrast)

# Create DataLoader to feed the model
train_dl_clean = DataLoader(train_ds_clean, batch_size=optimal_batch_size, shuffle=False)
train_dl_noise = DataLoader(train_ds_noise, batch_size=optimal_batch_size, shuffle=False)
visualize_test = DataLoader(test_ds, batch_size=1, shuffle=False)

# Initialise the model and train it with the full training set and with the optimal hyperparameters
final_model = Model()
test_loss, training_loss, autoencoder = final_model.fit(optimal_noise,
                                                        optimal_batch_size,
                                                        optimal_hidden_layers,
                                                        train_dl_clean,
                                                        train_dl_noise,
                                                        visualize_test,
                                                        optimal_noise_type,
                                                        optimal_epoch,
                                                        EPOCHS_PRETRAINING,
                                                        optimal_learning_rate)

# Save the optimal trained model using pickle
pickle.dump(autoencoder,open(f"Autoencoder_with"
      f"_BATCH_SIZE_{str(optimal_batch_size)}"
      f"_NOISE_TYPE_{optimal_noise_type}"
      f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
      f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
      f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
      f"_EPOCH_{str(optimal_epoch)}"
      f"_LOSS_{str(optimal_loss).replace('.', ',')}.sav", 'wb'))

# If we need to retrive the model we can use this
# autoencoder = pickle.load(open('final_autoencoder.sav', 'rb'))

# Save the lower dimensional training dataset as predicted by the optimal autoencoder
visualize_train = DataLoader(train_ds_clean, batch_size=1, shuffle=False)
reduced_train = np.zeros((len(visualize_train),13))
for i, features in enumerate(visualize_train):
    reduced_train[i] = autoencoder.encode(features[0]).detach().numpy()
np.savetxt(f"reduced_trainset_with"
      f"_BATCH_SIZE_{str(optimal_batch_size)}"
      f"_NOISE_TYPE_{optimal_noise_type}"
      f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
      f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
      f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
      f"_EPOCH_{str(optimal_epoch)}"
      f"_LOSS_{str(optimal_loss).replace('.', ',')}.csv", reduced_train, delimiter=',')

# Save the lower dimensional test dataset as predicted by the optimal autoencoder
reduced_test = np.zeros((len(visualize_test),13))
for i, features in enumerate(visualize_test):
    reduced_test[i] = autoencoder.encode(features[0]).detach().numpy()
np.savetxt(f"reduced_test_set_with"
      f"_BATCH_SIZE_{str(optimal_batch_size)}"
      f"_NOISE_TYPE_{optimal_noise_type}"
      f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
      f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
      f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
      f"_EPOCH_{str(optimal_epoch)}"
      f"_LOSS_{str(optimal_loss).replace('.', ',')}.csv", reduced_test, delimiter=',')

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

plt.savefig("final_test_prediction" + " with noise type " + optimal_noise_type +
          " [" + str(optimal_noise) + "], batch size " + str(optimal_batch_size) +
          " hidden layers " + ','.join([str(elem) for elem in optimal_hidden_layers]) + " lr " + str(optimal_learning_rate).replace('.', ',') +
          " epoch " + str(optimal_epoch) + " loss " + str(optimal_loss).replace('.', ','))

# Analyse the features that are captured by the first layer of the model
plt.figure(figsize=(30, 30))
for i in range(150):
    weights = autoencoder.encoders[0].detach().numpy()[:,i+3].reshape(28,28)
    ax = plt.subplot(15, 10, i + 1)
    plt.imshow(weights)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(f"features_captured_with"
      f"_BATCH_SIZE_{str(optimal_batch_size)}"
      f"_NOISE_TYPE_{optimal_noise_type}"
      f"_NOISE_PERCENTAGE_{str(optimal_noise).replace('.', ',')}"
      f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in optimal_hidden_layers])}]"
      f"_LEATNING_RATE_{str(optimal_learning_rate).replace('.', ',')}"
      f"_EPOCH_{str(optimal_epoch)}"
      f"_LOSS_{str(optimal_loss).replace('.', ',')}")