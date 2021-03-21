import pandas as pd
import numpy as np
import tensorflow as tf
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
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def fit(self,
              NOISE_PERCENTAGE,
              BATCH_SIZE,
              HIDDEN_LAYERS,
              train_dl_clean,
              train_dl_noise,
              test_dl,
              NOISE_TYPE="zeros",
              NUMBER_OF_PIXELS=784,
              GAUSSIAN_ST_DEV=None,
              EPOCHS_PRETRAINING=10,
              EPOCHS_FINETUNING=10):
        models = []
        visible_dim = NUMBER_OF_PIXELS
        dae_train_dl_clean = train_dl_clean
        dae_train_dl_corrupted = train_dl_noise
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
            dae_train_dl_corrupted = DataLoader(TensorDataset(torch.Tensor(new_data_corrupted)), batch_size=BATCH_SIZE,
                                                shuffle=False)
            visible_dim = hidden_dim
            epoch = 0

        # fine-tune autoencoder
        dae = DAE(models)
        optimizer = torch.optim.Adam(dae.parameters(), 1e-3)
        loss = nn.MSELoss()
        writer_train = SummaryWriter(f"./autoencoders_check_1_train"
                                     f"_BATCH_SIZE_{str(BATCH_SIZE)}"
                                     f"_NOISE_TYPE_{NOISE_TYPE}"
                                     f"_NOISE_PERCENTAGE_{str(NOISE_PERCENTAGE)}"
                                     f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in HIDDEN_LAYERS])}]")
        writer_validation = SummaryWriter(f"./autoencoders_check_1_validation"
                                     f"_BATCH_SIZE_{str(BATCH_SIZE)}"
                                     f"_NOISE_TYPE_{NOISE_TYPE}"
                                     f"_NOISE_PERCENTAGE_{str(NOISE_PERCENTAGE)}"
                                     f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in HIDDEN_LAYERS])}]")
        for epoch in range(EPOCHS_FINETUNING):
            print(epoch)
            ep_loss = 0
            val_ep_loss = 0
            for j, features in enumerate(test_dl):
                batch_loss = loss(features[0], dae(features[0]))
                val_ep_loss += batch_loss
            writer_validation.add_scalar("Loss", val_ep_loss/len(test_dl), epoch)
            for i, features in enumerate(train_dl_clean):
                batch_loss = loss(features[0], dae(features[0]))
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                ep_loss += batch_loss
            writer_train.add_scalar("Loss", ep_loss/len(train_dl_clean), epoch)
        plt.show()
