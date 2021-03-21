import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Autoencoders.DAE import DAE
from Autoencoders.d_DAE import d_DAE
from Autoencoders.utils import add_noise


# CONSTANTS
MAX_BRIGHTNESS = 255
MEAN = 0.5
NUMBER_OF_PIXELS = 784
PICTURE_DIMENSION = 28
BATCH_SIZE = 64
NOISE_PERCENTAGE = 0.1  #set it to "None" to impose gaussian noise
GAUSSIAN_ST_DEV = None   #set it to "None" to impose zero noise
HIDDEN_LAYERS = [500, 250, 100, 5]
EPOCHS_PRETRAINING = 10
EPOCHS_FINETUNING = 10

# import and extract the data
df = pd.read_csv("../Data/sign_mnist_train.csv")
df_test = pd.read_csv("../Data/sign_mnist_test.csv")
X_train = df.iloc[:,1:].values 
y_train = df.iloc[:,0].values
X_test = df_test.iloc[:,1:].values 
y_test = df_test.iloc[:,0].values

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
X_noise_zeros = np.zeros(np.shape(X_contrast))
for i in range(len(X_contrast)):
    X_noise_zeros[i] = add_noise(X_contrast[i, :], noise_type='zeros', percentage=NOISE_PERCENTAGE)
X_noise_zeros = torch.Tensor(X_noise_zeros)

X_noise_gaussian = np.zeros(np.shape(X_contrast))
for i in range(len(X_contrast)):
    X_noise_gaussian[i] = add_noise(X_contrast[i, :], noise_type='gaussian', sigma=GAUSSIAN_ST_DEV)
X_noise_gaussian = torch.Tensor(X_noise_gaussian)

train_ds_clean = TensorDataset(X_clean)
train_ds_zeros = TensorDataset(X_noise_zeros)
train_ds_gaussian = TensorDataset(X_noise_gaussian)
train_dl_clean = DataLoader(train_ds_clean, batch_size=BATCH_SIZE, shuffle=False)
train_dl_zeros = DataLoader(train_ds_zeros, batch_size=BATCH_SIZE, shuffle=False)
train_dl_gaussian = DataLoader(train_ds_gaussian, batch_size=BATCH_SIZE, shuffle=False)


models = []
visible_dim = NUMBER_OF_PIXELS
dae_train_dl_clean = train_dl_clean
if GAUSSIAN_ST_DEV is not None and NOISE_PERCENTAGE is not None:
    print("set either GAUSSIAN_ST_DEV or NOISE_PERCENTAGE to None")
if GAUSSIAN_ST_DEV is not None:
    dae_train_dl_corrupted = train_dl_gaussian
if NOISE_PERCENTAGE is not None:
    dae_train_dl_corrupted = train_dl_zeros
for hidden_dim in HIDDEN_LAYERS:

    # train d_DAE
    dae = d_DAE(visible_dim=visible_dim, hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(dae.parameters(), lr=0.01, weight_decay=1e-5)

    epochs = EPOCHS_PRETRAINING
    l = len(dae_train_dl_clean)
    losslist = list()
    epochloss = 0
    running_loss = 0
    dataset_previous_layer_batched = []
    for i, features in tqdm(enumerate(dae_train_dl_clean)):
        dataset_previous_layer_batched.append(features[0])

    for epoch in range(epochs):

        print("Entering Epoch: ", epoch)
        for i, features in tqdm(enumerate(dae_train_dl_corrupted)):
            # -----------------Forward Pass----------------------
            output = dae(features[0])
            loss = criterion(output, dataset_previous_layer_batched[i])
            # -----------------Backward Pass---------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epochloss += loss.item()
            # -----------------Log-------------------------------
    losslist.append(running_loss / l)
    running_loss = 0

    models.append(dae)
    # rederive new data loader based on hidden activations of trained model
    new_data = np.array([dae.encode(data_list[0])[0].detach().numpy() for data_list in dae_train_dl_corrupted])
    new_data_corrupted = np.zeros(np.shape(new_data))
    if GAUSSIAN_ST_DEV is not None:
        for i in range(len(new_data)):
            new_data_corrupted[i] = add_noise(new_data[i, :], noise_type='gaussian', sigma=GAUSSIAN_ST_DEV)

    if NOISE_PERCENTAGE is not None:
        for i in range(len(new_data)):
            new_data_corrupted[i] = add_noise(new_data[i, :], noise_type='zeros', percentage=NOISE_PERCENTAGE)
    # new_data= np.concatenate(new_data, axis=0)
    dae_train_dl_clean = DataLoader(TensorDataset(torch.Tensor(new_data)), batch_size=BATCH_SIZE, shuffle=False)
    dae_train_dl_corrupted = DataLoader(TensorDataset(torch.Tensor(new_data_corrupted)), batch_size=BATCH_SIZE, shuffle=False)
    visible_dim = hidden_dim
    epoch = 0

# fine-tune autoencoder
dae = DAE(models)
optimizer = torch.optim.Adam(dae.parameters(), 1e-3)
loss = nn.MSELoss()
ep_loss = 0
for epoch in range(EPOCHS_FINETUNING):
    print(epoch)
    for i, features in enumerate(train_dl_clean):
        batch_loss = loss(features[0], dae(features[0]))
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        ep_loss += batch_loss

X_test = torch.Tensor(X_test)
test_ds = TensorDataset(X_test)
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
    plt.imshow(dae(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 9:
        break
plt.savefig('autoencoder_pictures.pdf')
plt.show()