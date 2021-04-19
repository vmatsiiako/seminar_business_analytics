import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from Autoencoders.finetuning_AE import FinetuningAE
from Autoencoders.pretraining_DAE import PretrainingDAE
from Autoencoders.utils import add_noise


class DeepAutoencoder(nn.Module):
    """
    A class used to represent the Deep Autoencoder model

    Methods
    -------
    fit(pretraining_noise_type,
        pretraining_noise_parameter,
        batch_size,
        hidden_layers,
        training_set_clean,
        validation_set_clean,
        epochs_finetuning,
        epochs_pretraining,
        learning_rate_pretraining,
        learning_rate_finetuning,
        number_pixels = 784)

        Trains the Denoising Deep Autoencoder and returns the training/validation
        vector losses together with the tuned autoencoder model.

    """
    def __init__(self):
        super(DeepAutoencoder, self).__init__()

    def fit(self,
            pretraining_noise_type,
            pretraining_noise_parameter,
            batch_size,
            hidden_layers,
            training_set_clean,
            validation_set_clean,
            epochs_finetuning,
            epochs_pretraining,
            learning_rate_pretraining,
            learning_rate_finetuning,
            input_nodes=784):
        """Trains the Deep Autoencoder and returns the training/validation vector
        losses together with  the tuned autoencoder model

        Parameters
        ----------
        pretraining_noise_type : str
            The type of noise to be used in the pretraining phase
        pretraining_noise_parameter : float
            The parameter of the noise used for the pretraining phase
        batch_size : int
            batch size used for both pretraining and finetuning (usually powers of 2)
        hidden_layers : List[int]
            The array where each entry represents the number of nodes for
            each encoder layer. The decoder is the mirror
        training_set_clean : ndarray
            The numpy array where each row is a training observation
        validation_set_clean : ndarray
            The numpy array used as validation
        epochs_finetuning : int
            The number of epochs used for the finetuning phase
        epochs_pretraining : int
            The number of epochs used for the pretraining phase
        learning_rate_pretraining : float
            The learning rate used in the pretraining optimization
        learning_rate_finetuning : float
            The learning rate used in the finetuning optimization
        input_nodes : int
            The number of original features, that is the number of nodes
            for the input layer (default is 784)

        Returns
        -------
        float
            the validation loss for this batch
        float
            the training loss for this batch
        DeepAutoencoder
            the resulting trained DeepAutoencoder model
        """

        # Initialize the list of pretrained models to an empty list
        models = []
        visible_dim = input_nodes

        # Initialize the data that are used to train the stacked denoising autoencoders to be the original observations
        pretrain_data_clean = training_set_clean

        # Pre-training - Loop over all hidden layers
        for hidden_dim in hidden_layers:

            # Initialize the denoising autoencoder, the loss criterion and the optimizer
            pretraining_dae = PretrainingDAE(visible_dim=visible_dim, hidden_dim=hidden_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(pretraining_dae.parameters(), lr=learning_rate_pretraining, weight_decay=1e-5)

            epoch_loss = 0
            running_loss = 0

            # Pretraining epochs
            for epoch in range(epochs_pretraining):
                # Construct the corrupted version of the dataset dataset
                pretrain_data_noise = np.zeros(np.shape(pretrain_data_clean))
                for i in range(len(pretrain_data_clean)):
                    pretrain_data_noise[i] = add_noise(pretrain_data_clean[i, :], noise_type=pretraining_noise_type,
                                                       parameter=pretraining_noise_parameter)

                # Transform clean and noised numpy arrays into Tensors
                pretrain_clean_tensor = torch.Tensor(pretrain_data_clean)
                pretrain_noise_tensor = torch.Tensor(pretrain_data_noise)

                # Construct the clean and noised Tensor Datasets
                pretrain_ds_clean = TensorDataset(pretrain_clean_tensor)
                pretrain_ds_noise = TensorDataset(pretrain_noise_tensor)

                # Create the clean and noised DataLoaders
                dae_train_dl_clean = DataLoader(pretrain_ds_clean, batch_size=batch_size, shuffle=False)
                dae_train_dl_corrupted = DataLoader(pretrain_ds_noise, batch_size=batch_size, shuffle=False)

                # Create an iterator for the clean DataLoader to be able to access its elements using indexes
                dataloader_iterator = iter(dae_train_dl_clean)

                # Print the epoch to keep truck of the pretraining process
                print("Pretraining Epoch #", epoch)

                # Loop over all batches
                for i, features in tqdm(enumerate(dae_train_dl_corrupted)):
                    # Forward pass
                    output = pretraining_dae(features[0])

                    # Compute the loss comparing the outputs of the autoencoder to the clean input
                    loss = criterion(output, next(dataloader_iterator)[0])

                    # Backward Pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update the losses
                    running_loss += loss.item()
                    epoch_loss += loss.item()

            # Append the pretrained weights and biases
            models.append(pretraining_dae)

            # Derive the input for the following hidden layer based on hidden activations of trained model
            pretrain_data_clean = np.array([pretraining_dae.encode(data_list[0])[0].detach().numpy() for data_list in dae_train_dl_corrupted])

            # Update the visible/input dimension for the new layer
            visible_dim = hidden_dim

        # Initialize the final Deep Autoencoder using the pre-trained wights amd biases
        final_AE = FinetuningAE(models)
        optimizer = torch.optim.Adam(final_AE.parameters(), lr=learning_rate_finetuning)
        loss = nn.MSELoss()

        # Initialize the vector to store the validation and training losses
        val_loss = np.zeros(epochs_finetuning)
        final_train_loss = np.zeros(epochs_finetuning)

        # Transform the clean input and the validation set into Tensors
        trainign_clean_tensor = torch.Tensor(training_set_clean)
        validation_clean_tensor = torch.Tensor(validation_set_clean)

        # Construct the clean input and validation Tensor Datasets
        training_ds_clean = TensorDataset(trainign_clean_tensor)
        validation_ds = TensorDataset(validation_clean_tensor)

        # Create the clean and noised DataLoaders
        training_dl_clean = DataLoader(training_ds_clean, batch_size=batch_size, shuffle=False)
        validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

        # Fine-tuning - Loop over all fine-tuning epochs
        for epoch in range(epochs_finetuning):
            # Print the epoch to keep truck of the training process
            print(f"Fine_tuning_Epoch{str(epoch)}")

            epoch_loss = 0
            validation_epoch_loss = 0
            final_training_loss = 0

            # Loop over all batches
            for i, features in enumerate(training_dl_clean):
                # Forward pass
                output = final_AE(features[0])

                # Compute the loss comparing the outputs of the autoencoder to the clean input
                batch_loss = loss(features[0], output)

                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # Update the epoch loss
                epoch_loss += batch_loss

            # Loop over all batches of the training set to compute the current training loss
            for k, features in enumerate(training_dl_clean):
                # Compute the loss after doing a forward pass over the autoencoder
                batch_loss = loss(features[0], final_AE(features[0]))

                # Update the training loss
                final_training_loss += batch_loss

            # Store the training loss for the current epoch
            final_train_loss[epoch] = final_training_loss/len(training_dl_clean)

            # Loop over all batches of the validation set to compute the current validation loss
            for j, features in enumerate(validation_dl):
                # Compute the loss after doing a forward pass over the autoencoder
                batch_loss = loss(features[0], final_AE(features[0]))

                # Update the validation loss
                validation_epoch_loss += batch_loss

            # Store the validation loss for the current epoch
            val_loss[epoch] = validation_epoch_loss/len(validation_dl)

            # print the current training and validation loss to keep track of the training process
            print(f"train loss = {str(final_train_loss[epoch])}"
                  f", validation loss = {str(val_loss[epoch])}")

        return val_loss, final_train_loss, final_AE


class DenoisingDeepAutoencoder(nn.Module):
    """
    A class used to represent the Denoising Deep Autoencoder model

    Methods
    -------
    fit(pretraining_noise_type,
        pretraining_noise_parameter,
        finetuning_noise_type,
        fientuning_noise_parameter,
        batch_size,
        hidden_layers,
        training_set_clean,
        validation_set_clean,
        epochs_finetuning,
        epochs_pretraining,
        learning_rate_pretraining,
        learning_rate_finetuning,
        number_pixels = 784)

        Trains the Denoising Deep Autoencoder and returns the training/validation
        vector losses together with the tuned autoencoder model.

    """
    def __init__(self):
        super(DenoisingDeepAutoencoder, self).__init__()

    def fit(self,
            pretraining_noise_type,
            pretraining_noise_parameter,
            finetuning_noise_type,
            finetuning_noise_parameter,
            batch_size,
            hidden_layers,
            training_set_clean,
            validation_set_clean,
            epochs_finetuning,
            epochs_pretraining,
            learning_rate_pretraining,
            learning_rate_finetuning,
            input_nodes=784):
        """Trains the Denoising Deep Autoencoder and returns the training/validation
        vector losses together with  the tuned autoencoder model

        Parameters
        ----------
        pretraining_noise_type : str
            The type of noise to be used in the pretraining phase
        pretraining_noise_parameter : float
            The parameter of the noise used for the pretraining phase
        finetuning_noise_type : str
            The type of noise to be used in the finetuning phase
        finetuning_noise_parameter : float
            The parameter of the noise used for the finetuning phase
        batch_size : int
            batch size used for both pretraining and finetuning (usually powers of 2)
        hidden_layers : List[int]
            The array where each entry represents the number of nodes for
            each encoder layer. The decoder is the mirror
        training_set_clean : ndarray
            The numpy array where each row is a training observation
        validation_set_clean : ndarray
            The numpy array used as validation
        epochs_finetuning : int
            The number of epochs used for the finetuning phase
        epochs_pretraining : int
            The number of epochs used for the pretraining phase
        learning_rate_pretraining : float
            The learning rate used in the pretraining optimization
        learning_rate_finetuning : float
            The learning rate used in the finetuning optimization
        input_nodes : int
            The number of original features, that is the number of nodes
            for the input layer (default is 784)

        Returns
        -------
        float
            the validation loss for this batch
        float
            the training loss for this batch
        DenoisingDeepAutoencoder
            the resulting trained DenoisingDeepAutoencoder model
        """

        # Initialize the list of pretrained models to an empty list
        models = []
        visible_dim = input_nodes

        # Initialize the data that are used to train the stacked denoising autoencoders to be the original observations
        pretrain_data_clean = training_set_clean

        # Pre-training - Loop over all hidden layers
        for hidden_dim in hidden_layers:

            # Initialize the denoising autoencoder, the loss criterion and the optimizer
            dae = PretrainingDAE(visible_dim=visible_dim, hidden_dim=hidden_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(dae.parameters(), lr=learning_rate_pretraining, weight_decay=1e-5)

            epoch_loss = 0
            running_loss = 0

            # Pretraining epochs
            for epoch in range(epochs_pretraining):
                # Construct the corrupted version of the dataset dataset
                pretrain_data_noise = np.zeros(np.shape(pretrain_data_clean))
                for i in range(len(pretrain_data_clean)):
                    pretrain_data_noise[i] = add_noise(pretrain_data_clean[i, :], noise_type=pretraining_noise_type,
                                                       parameter=pretraining_noise_parameter)

                # Transform clean and noised numpy arrays into Tensors
                pretrain_clean_tensor = torch.Tensor(pretrain_data_clean)
                pretrain_noise_tensor = torch.Tensor(pretrain_data_noise)

                # Construct the clean and noised Tensor Datasets
                pretrain_ds_clean = TensorDataset(pretrain_clean_tensor)
                pretrain_ds_noise = TensorDataset(pretrain_noise_tensor)

                # Create the clean and noised DataLoaders
                dae_train_dl_clean = DataLoader(pretrain_ds_clean, batch_size=batch_size, shuffle=False)
                dae_train_dl_corrupted = DataLoader(pretrain_ds_noise, batch_size=batch_size, shuffle=False)

                # Create an iterator for the clean DataLoader to be able to access its elements using indexes
                dataloader_iterator = iter(dae_train_dl_clean)

                # Print the epoch to keep truck of the pretraining process
                print("Pretraining Epoch #", epoch)

                # Loop over all batches
                for i, features in tqdm(enumerate(dae_train_dl_corrupted)):
                    # Forward pass
                    output = dae(features[0])

                    # Compute the loss comparing the outputs of the autoencoder to the clean input
                    loss = criterion(output, next(dataloader_iterator)[0])

                    # Backward Pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update the losses
                    running_loss += loss.item()
                    epoch_loss += loss.item()

            # Append the pretrained weights and biases
            models.append(dae)

            # Derive the input for the following hidden layer based on hidden activations of trained model
            pretrain_data_clean = np.array([dae.encode(data_list[0])[0].detach().numpy() for data_list in dae_train_dl_corrupted])

            # Update the visible/input dimension for the new layer
            visible_dim = hidden_dim

        # Initialize the final Denoising Deep Autoencoder using the pre-trained wights amd biases
        final_DAE = FinetuningAE(models)
        optimizer = torch.optim.Adam(final_DAE.parameters(), lr=learning_rate_finetuning)
        loss = nn.MSELoss()

        # Initialize the vector to store the validation and training losses
        val_loss = np.zeros(epochs_finetuning)
        final_train_loss = np.zeros(epochs_finetuning)

        # Transform the clean input and the validation set into Tensors
        trainign_clean_tensor = torch.Tensor(training_set_clean)
        validation_clean_tensor = torch.Tensor(validation_set_clean)

        # Construct the clean input and validation Tensor Datasets
        training_ds_clean = TensorDataset(trainign_clean_tensor)
        validation_ds = TensorDataset(validation_clean_tensor)

        # Create the clean and noised DataLoaders
        training_dl_clean = DataLoader(training_ds_clean, batch_size=batch_size, shuffle=False)
        validation_dl = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

        # Fine-tuning - Loop over all fine-tuning epochs
        for epoch in range(epochs_finetuning):
            # Print the epoch to keep truck of the training process
            print(f"Fine_tuning_Epoch{str(epoch)}")

            # Construct the noised dataset
            X_finetuning_noise = np.zeros(np.shape(training_set_clean))
            for i in range(len(training_set_clean)):
                X_finetuning_noise[i, :] = add_noise(training_set_clean[i, :], noise_type=finetuning_noise_type,
                                                     parameter=finetuning_noise_parameter)
            X_finetuning_noise = torch.Tensor(X_finetuning_noise)

            # Construct the noised Data Loader
            finetuning_ds_noise = TensorDataset(X_finetuning_noise)
            finetuning_dl_noise = DataLoader(finetuning_ds_noise, batch_size=batch_size, shuffle=False)

            epoch_loss = 0
            validation_epoch_loss = 0
            final_training_loss = 0

            # Create an iterator for the clean DataLoader to be able to access its elements using indexes
            dataloader_iterator_ae = iter(training_dl_clean)

            # Loop over all batches
            for i, features in enumerate(finetuning_dl_noise):
                # Forward pass
                output = final_DAE(features[0])

                # Compute the loss comparing the outputs of the autoencoder to the clean input
                batch_loss = loss(output, next(dataloader_iterator_ae)[0])

                # Backward pass
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # Update the epoch loss
                epoch_loss += batch_loss

            # Loop over all batches of the training set to compute the current training loss
            for k, features in enumerate(training_dl_clean):
                # Compute the loss after doing a forward pass over the autoencoder
                batch_loss = loss(features[0], final_DAE(features[0]))

                # Update the training loss
                final_training_loss += batch_loss

            # Store the training loss for the current epoch
            final_train_loss[epoch] = final_training_loss/len(training_dl_clean)

            # Loop over all batches of the validation set to compute the current validation loss
            for j, features in enumerate(validation_dl):
                # Compute the loss after doing a forward pass over the autoencoder
                batch_loss = loss(features[0], final_DAE(features[0]))

                # Update the validation loss
                validation_epoch_loss += batch_loss

            # Store the validation loss for the current epoch
            val_loss[epoch] = validation_epoch_loss/len(validation_dl)

            # print the current training and validation loss to keep track of the training process
            print(f"train loss = {str(final_train_loss[epoch])}"
                  f", validation loss = {str(val_loss[epoch])}")

        return val_loss, final_train_loss, final_DAE
