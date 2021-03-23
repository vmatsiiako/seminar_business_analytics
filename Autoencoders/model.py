import numpy as np
# import datetime.datetime as dt
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
              validation_dl,
              NOISE_TYPE="zeros",
              NUMBER_OF_PIXELS=784,
              GAUSSIAN_ST_DEV=None,
              EPOCHS_PRETRAINING=2,
              EPOCHS_FINETUNING=2):
        models = []
        visible_dim = NUMBER_OF_PIXELS
        dae_train_dl_clean = train_dl_clean
        dae_train_dl_corrupted = train_dl_noise
        for hidden_dim in HIDDEN_LAYERS:

            # train d_DAE
            dae = d_DAE(visible_dim=visible_dim, hidden_dim=hidden_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(dae.parameters(), lr=0.01, weight_decay=1e-5)

            l = len(dae_train_dl_clean)
            losslist = list()
            epoch_loss = 0
            running_loss = 0
            dataset_previous_layer_batched = []
            for i, features in tqdm(enumerate(dae_train_dl_clean)):
                dataset_previous_layer_batched.append(features[0])

            for epoch in range(EPOCHS_PRETRAINING):

                print("Pretraining Epoch #", epoch)
                for i, features in tqdm(enumerate(dae_train_dl_corrupted)):
                    # -----------------Forward Pass----------------------
                    output = dae(features[0])
                    loss = criterion(output, dataset_previous_layer_batched[i])
                    # -----------------Backward Pass---------------------
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    # -----------------Log-------------------------------
            losslist.append(running_loss / l)

            models.append(dae)
            # rederive new data loader based on hidden activations of trained model
            new_data = np.array([dae.encode(data_list[0])[0].detach().numpy() for data_list in dae_train_dl_corrupted])
            new_data_corrupted = np.zeros(np.shape(new_data))

            for i in range(len(new_data)):
                new_data_corrupted[i] = add_noise(new_data[i, :], noise_type=NOISE_TYPE, percentage=NOISE_PERCENTAGE)
            new_data_corrupted = torch.Tensor(new_data_corrupted)
            dae_train_dl_clean = DataLoader(TensorDataset(torch.Tensor(new_data)), batch_size=BATCH_SIZE, shuffle=False)
            dae_train_dl_corrupted = DataLoader(TensorDataset(torch.Tensor(new_data_corrupted)), batch_size=BATCH_SIZE,
                                                shuffle=False)
            visible_dim = hidden_dim

        # fine-tune autoencoder
        ae = DAE(models)
        optimizer = torch.optim.Adam(ae.parameters(), 1e-3)
        loss = nn.MSELoss()
        writer_train = SummaryWriter(f"./autoencoders_check_1_train"
                                     f"_BATCH_SIZE_{str(BATCH_SIZE)}"
                                     f"_NOISE_TYPE_{NOISE_TYPE}"
                                     f"_NOISE_PERCENTAGE_{str(NOISE_PERCENTAGE)}"
                                     f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in HIDDEN_LAYERS])}]"
                                     #f"_TIME_{dt.now()}"
                                     )
        writer_validation = SummaryWriter(f"./autoencoders_check_1_validation"
                                          f"_BATCH_SIZE_{str(BATCH_SIZE)}"
                                          f"_NOISE_TYPE_{NOISE_TYPE}"
                                          f"_NOISE_PERCENTAGE_{str(NOISE_PERCENTAGE)}"
                                          f"_HIDDEN_LAYERS_[{','.join([str(elem) for elem in HIDDEN_LAYERS])}]"
                                          #f"_TIME_{dt.now()}"
                                          )
        val_loss = []
        train_loss = []
        for epoch in range(EPOCHS_FINETUNING):
            print("Fine-tuning Epoch #" + str(epoch))
            epoch_loss = 0
            validation_epoch_loss = 0
            for j, features in enumerate(validation_dl):
                batch_loss = loss(features[0], ae(features[0]))
                validation_epoch_loss += batch_loss
            val_loss.append(validation_epoch_loss)
            writer_validation.add_scalar("Loss", validation_epoch_loss/len(validation_dl), epoch)
            for i, features in enumerate(train_dl_clean):
                batch_loss = loss(features[0], ae(features[0]))
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            train_loss.append(epoch_loss)
            writer_train.add_scalar("Loss", epoch_loss/len(train_dl_clean), epoch)
        plt.show()

        return val_loss, train_loss, ae

