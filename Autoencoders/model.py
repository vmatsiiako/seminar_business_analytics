import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Autoencoders.DAE import DAE
from Autoencoders.d_DAE import d_DAE
from Autoencoders.utils import add_noise

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def fit(self,
            PRETRAINING_NOISE_TYPE,
            PRETRAINING_NOISE_PARAMETER,
            FINETUNING_NOISE_TYPE,
            FINETUNING_NOISE_PARAMETER,
            BATCH_SIZE,
            HIDDEN_LAYERS,
            X_train_clean,
            X_validation_clean,
            EPOCHS_FINETUNING,
            EPOCHS_PRETRAINING,
            LEARNING_RATE,
            NUMBER_OF_PIXELS=784):

        # construct the noised train set for pretraining
        # X_pretrain_noise = np.zeros(np.shape(X_train_clean))
        # for i in range(len(X_train_clean)):
        #     X_pretrain_noise[i] = add_noise(X_train_clean[i, :], noise_type=PRETRAINING_NOISE_TYPE, parameter=PRETRAINING_NOISE_PARAMETER)
        # X_train_clean_tensor = torch.Tensor(X_train_clean)
        # X_pretrain_noise = torch.Tensor(X_pretrain_noise)
        #
        # ds_clean = TensorDataset(X_train_clean_tensor)
        # pretrain_ds_noise = TensorDataset(X_pretrain_noise)
        # dl_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False)
        # pretrain_dl_noise = DataLoader(pretrain_ds_noise, batch_size=BATCH_SIZE, shuffle=False)

        # train_ds_clean = TensorDataset(X_train_clean)
        # train_ds_noise = TensorDataset(X_train_noise)
        # validation_ds = TensorDataset(X_validation_clean)
        # train_dl_clean = DataLoader(train_ds_clean, batch_size=BATCH_SIZE, shuffle=False)
        # train_dl_noise = DataLoader(train_ds_noise, batch_size=BATCH_SIZE, shuffle=False)
        # validation_dl = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False)
        models = []
        visible_dim = NUMBER_OF_PIXELS
        new_data = X_train_clean
        # dae_train_dl_corrupted = pretrain_dl_noise
        for hidden_dim in HIDDEN_LAYERS:

            # train d_DAE
            dae = d_DAE(visible_dim=visible_dim, hidden_dim=hidden_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(dae.parameters(), lr=0.01, weight_decay=1e-5)

            # l = len(dae_train_dl_clean)
            # losslist = list()
            epoch_loss = 0
            running_loss = 0
            # dataset_previous_layer_batched = []
            # for i, features in tqdm(enumerate(dae_train_dl_clean)):
            #     dataset_previous_layer_batched.append(features[0])

            for epoch in range(EPOCHS_PRETRAINING):
                # construct the noised train set for pretraining
                X_pretrain_noise = np.zeros(np.shape(new_data))
                for i in range(len(new_data)):
                    X_pretrain_noise[i] = add_noise(new_data[i, :], noise_type=PRETRAINING_NOISE_TYPE,
                                                    parameter=PRETRAINING_NOISE_PARAMETER)
                X_train_clean_tensor = torch.Tensor(new_data)
                X_pretrain_noise = torch.Tensor(X_pretrain_noise)

                ds_clean = TensorDataset(X_train_clean_tensor)
                pretrain_ds_noise = TensorDataset(X_pretrain_noise)
                dae_train_dl_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False)
                dae_train_dl_corrupted = DataLoader(pretrain_ds_noise, batch_size=BATCH_SIZE, shuffle=False)

                # dae_train_dl_clean = dl_clean
                # dae_train_dl_corrupted = pretrain_dl_noise

                dataloader_iterator = iter(dae_train_dl_clean)
                print("Pretraining Epoch #", epoch)
                for i, features in tqdm(enumerate(dae_train_dl_corrupted)):
                    # -----------------Forward Pass----------------------
                    output = dae(features[0])
                    # loss = criterion(output, dataset_previous_layer_batched[i])
                    loss = criterion(output, next(dataloader_iterator)[0])
                    # -----------------Backward Pass---------------------
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    # -----------------Log-------------------------------
            # losslist.append(running_loss / l)

            models.append(dae)
            # rederive new data loader based on hidden activations of trained model
            new_data = np.array([dae.encode(data_list[0])[0].detach().numpy() for data_list in dae_train_dl_corrupted])

            # new_data_corrupted = np.zeros(np.shape(new_data))
            #
            # for i in range(len(new_data)):
            #     new_data_corrupted[i] = add_noise(new_data[i, :], noise_type=PRETRAINING_NOISE_TYPE, parameter=PRETRAINING_NOISE_PARAMETER)
            #
            # new_data_corrupted = torch.Tensor(new_data_corrupted)
            # dae_train_dl_clean = DataLoader(TensorDataset(torch.Tensor(new_data)), batch_size=BATCH_SIZE, shuffle=False)
            # dae_train_dl_corrupted = DataLoader(TensorDataset(torch.Tensor(new_data_corrupted)), batch_size=BATCH_SIZE,
            #                                     shuffle=False)
            visible_dim = hidden_dim

        # fine-tune autoencoder
        ae = DAE(models)
        optimizer = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)
        loss = nn.MSELoss()

        # construct the noised train set for pretraining
        # X_finetuning_noise = np.zeros(np.shape(X_train_clean))
        # for i in range(len(X_train_clean)):
        #     X_finetuning_noise[i] = add_noise(X_train_clean[i, :], noise_type=FINETUNING_NOISE_TYPE,
        #                                     parameter=FINETUNING_NOISE_PARAMETER)
        # X_finetuning_noise = torch.Tensor(X_finetuning_noise)
        #
        # finetuning_ds_noise = TensorDataset(X_finetuning_noise)
        # finetuning_dl_noise = DataLoader(finetuning_ds_noise, batch_size=BATCH_SIZE, shuffle=False)
        # X_validation_clean = torch.Tensor(X_validation_clean)
        # validation_ds = TensorDataset(X_validation_clean)
        # validation_dl = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False)

        val_loss = np.zeros((EPOCHS_FINETUNING))
        final_train_loss = np.zeros((EPOCHS_FINETUNING))
        # final_train_loss = []

        X_train_clean_tensor = torch.Tensor(X_train_clean)
        ds_clean = TensorDataset(X_train_clean_tensor)
        dl_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False)

        X_validation_clean = torch.Tensor(X_validation_clean)
        validation_ds = TensorDataset(X_validation_clean)
        validation_dl = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False)

        for epoch in range(EPOCHS_FINETUNING):
            print(f"Fine_tuning_Epoch{str(epoch)}")

            X_finetuning_noise = np.zeros(np.shape(X_train_clean))
            for i in range(len(X_train_clean)):
                X_finetuning_noise[i] = add_noise(X_train_clean[i, :], noise_type=FINETUNING_NOISE_TYPE,
                                                  parameter=FINETUNING_NOISE_PARAMETER)
            X_finetuning_noise = torch.Tensor(X_finetuning_noise)

            finetuning_ds_noise = TensorDataset(X_finetuning_noise)
            finetuning_dl_noise = DataLoader(finetuning_ds_noise, batch_size=BATCH_SIZE, shuffle=False)


            epoch_loss = 0
            validation_epoch_loss = 0
            final_training_loss = 0
            dataloader_iterator_ae = iter(dl_clean)   #comment out to not have noise
            # for i, features in enumerate(train_dl_clean):
            for i, features in enumerate(finetuning_dl_noise):   #comment out to not have noise
                output = ae(features[0])    #comment out to not have noise
                # batch_loss = loss(features[0], ae(features[0]))
                batch_loss = loss(output, next(dataloader_iterator_ae)[0])  #comment out to not have noise
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss
            for k, features in enumerate(dl_clean):
                batch_loss = loss(features[0], ae(features[0]))
                final_training_loss += batch_loss
            final_train_loss[epoch] = final_training_loss/len(dl_clean)
            for j, features in enumerate(validation_dl):
                batch_loss = loss(features[0], ae(features[0]))
                validation_epoch_loss += batch_loss
            val_loss[epoch] = validation_epoch_loss/len(validation_dl)
            print(f"train loss = {str(final_train_loss[epoch])}"
                  f", validation loss = {str(val_loss[epoch])}")
        # plt.show()

        return val_loss, final_train_loss, ae
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#
#     def fit(self,
#               NOISE_PARAMETER,
#               BATCH_SIZE,
#               HIDDEN_LAYERS,
#               train_dl_clean,
#               train_dl_noise,
#               validation_dl,
#               # X_train_clean,
#               # X_train_noise,
#               # X_validation_clean,
#               NOISE_TYPE,
#               EPOCHS_FINETUNING,
#               EPOCHS_PRETRAINING,
#               LEARNING_RATE,
#               NUMBER_OF_PIXELS=784):
#
#         # train_ds_clean = TensorDataset(X_train_clean)
#         # train_ds_noise = TensorDataset(X_train_noise)
#         # validation_ds = TensorDataset(X_validation_clean)
#         # train_dl_clean = DataLoader(train_ds_clean, batch_size=BATCH_SIZE, shuffle=False)
#         # train_dl_noise = DataLoader(train_ds_noise, batch_size=BATCH_SIZE, shuffle=False)
#         # validation_dl = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False)
#         models = []
#         visible_dim = NUMBER_OF_PIXELS
#         dae_train_dl_clean = train_dl_clean
#         dae_train_dl_corrupted = train_dl_noise
#         for hidden_dim in HIDDEN_LAYERS:
#
#             # train d_DAE
#             dae = d_DAE(visible_dim=visible_dim, hidden_dim=hidden_dim)
#             criterion = nn.MSELoss()
#             optimizer = torch.optim.Adam(dae.parameters(), lr=0.01, weight_decay=1e-5)
#
#             l = len(dae_train_dl_clean)
#             # losslist = list()
#             epoch_loss = 0
#             running_loss = 0
#             # dataset_previous_layer_batched = []
#             # for i, features in tqdm(enumerate(dae_train_dl_clean)):
#             #     dataset_previous_layer_batched.append(features[0])
#
#             for epoch in range(EPOCHS_PRETRAINING):
#                 dataloader_iterator = iter(dae_train_dl_clean)
#                 print("Pretraining Epoch #", epoch)
#                 for i, features in tqdm(enumerate(dae_train_dl_corrupted)):
#                     # -----------------Forward Pass----------------------
#                     output = dae(features[0])
#                     # loss = criterion(output, dataset_previous_layer_batched[i])
#                     loss = criterion(output, next(dataloader_iterator)[0])
#                     # -----------------Backward Pass---------------------
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#                     running_loss += loss.item()
#                     epoch_loss += loss.item()
#                     # -----------------Log-------------------------------
#             # losslist.append(running_loss / l)
#
#             models.append(dae)
#             # rederive new data loader based on hidden activations of trained model
#             new_data = np.array([dae.encode(data_list[0])[0].detach().numpy() for data_list in dae_train_dl_corrupted])
#             new_data_corrupted = np.zeros(np.shape(new_data))
#
#             for i in range(len(new_data)):
#                 new_data_corrupted[i] = add_noise(new_data[i, :], noise_type=NOISE_TYPE, parameter=NOISE_PARAMETER)
#             new_data_corrupted = torch.Tensor(new_data_corrupted)
#             dae_train_dl_clean = DataLoader(TensorDataset(torch.Tensor(new_data)), batch_size=BATCH_SIZE, shuffle=False)
#             dae_train_dl_corrupted = DataLoader(TensorDataset(torch.Tensor(new_data_corrupted)), batch_size=BATCH_SIZE,
#                                                 shuffle=False)
#             visible_dim = hidden_dim
#
#         # fine-tune autoencoder
#         ae = DAE(models)
#         optimizer = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)
#         loss = nn.MSELoss()
#
#         val_loss = np.zeros((EPOCHS_FINETUNING))
#         final_train_loss = np.zeros((EPOCHS_FINETUNING))
#         # final_train_loss = []
#
#         for epoch in range(EPOCHS_FINETUNING):
#             print(f"Fine_tuning_Epoch{str(epoch)}")
#             epoch_loss = 0
#             validation_epoch_loss = 0
#             final_training_loss = 0
#             dataloader_iterator_ae = iter(train_dl_clean)   #comment out to not have noise
#             # for i, features in enumerate(train_dl_clean):
#             for i, features in enumerate(train_dl_noise):   #comment out to not have noise
#                 output = ae(features[0])    #comment out to not have noise
#                 # batch_loss = loss(features[0], ae(features[0]))
#                 batch_loss = loss(output, next(dataloader_iterator_ae)[0])  #comment out to not have noise
#                 optimizer.zero_grad()
#                 batch_loss.backward()
#                 optimizer.step()
#                 epoch_loss += batch_loss
#             for k, features in enumerate(train_dl_clean):
#                 batch_loss = loss(features[0], ae(features[0]))
#                 final_training_loss += batch_loss
#             final_train_loss[epoch] = final_training_loss/len(train_dl_clean)
#             for j, features in enumerate(validation_dl):
#                 batch_loss = loss(features[0], ae(features[0]))
#                 validation_epoch_loss += batch_loss
#             val_loss[epoch] = validation_epoch_loss/len(validation_dl)
#             print(f"train loss = {str(final_train_loss[epoch])}"
#                   f", validation loss = {str(val_loss[epoch])}")
#         # plt.show()
#
#         return val_loss, final_train_loss, ae
#
