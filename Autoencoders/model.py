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

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/asl')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def fit(self,
              NOISE_PERCENTAGE,
              BATCH_SIZE,
              HIDDEN_LAYERS,
              train_dl_clean,
              train_dl_noise,
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
        ep_loss = 0
        for epoch in range(EPOCHS_FINETUNING):
            print(epoch)
            for i, features in enumerate(train_dl_clean):
                batch_loss = loss(features[0], dae(features[0]))
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                ep_loss += batch_loss

                running_loss += batch_loss.item()
                writer.add_scalar('training loss',
                                  running_loss,
                                  epoch * len(train_dl_clean) + i)

            running_loss = 0.0

        # # construct a plot that plots and saves the training history
        # N = np.arange(0, EPOCHS_FINETUNING)
        # plt.style.use("ggplot")
        # plt.figure()
        # plt.plot(N, dae.history["loss"], label="train_loss")
        # plt.plot(N, dae.history["val_loss"], label="val_loss")
        # plt.title("Training Loss and Accuracy")
        # plt.xlabel("Epoch #")
        # plt.ylabel("Loss/Accuracy")
        # plt.legend(loc="lower left")
        # plt.savefig(f'autoencoder_pictures_{str(BATCH_SIZE)}'
        #             f'_Gaussian_{str(GAUSSIAN_ST_DEV)}'
        #             f'_Zeros_{str(NOISE_PERCENTAGE)}'
        #             f'_Pretrain_{str(EPOCHS_PRETRAINING)}'
        #             f'_Layers_{str(HIDDEN_LAYERS)}.pdf')

        plt.show()
