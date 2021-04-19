import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def add_noise(img, noise_type, parameter=None):
    """Gets the original observation (image) and returns the corrupted version of it
    using the noise selected by the researcher

    Parameters
    ----------
    img : nparray
        The original observation represented by the array of 784 pixels
    noise_type: str
        The type of noise selected by the researcher
    parameter: float
        The parameter selected by the researcher to be used as parameter for the nosie type used.
        If the noise type is 'zeros', the parameter represents the percentage of pixels set to zero.
        If the noise type is 'gaussian', the parameter is the st. dev. of the normal distribution
        from which the added noise is drawn.

    Returns
    -------
    nparray
        The array of 784 pixels representing the corrupted image
    """

    img = img.astype(np.float32)

    if noise_type == "gaussian":
        mean = 0
        noise = np.random.normal(mean, parameter, img.shape)
        img = img + noise
        return img

    if noise_type == "zeros":
        noise = np.random.choice([0, 1], size=img.shape, p=[parameter, 1 - parameter])
        img = img * noise
        return img


def create_title(batch_size, pretraining_noise_type, pretraining_noise_parameter, hidden_layers,
                 learning_rate_pretraining, learning_rate_finetuning, epoch_pretraining=None, epoch_finetuning=None,
                 finetuning_noise_type=None, finetuning_noise_parameter=None, loss=None):
    """Creates the title of pictures or other files to be saved given certain inputs

    Parameters
    ----------
    batch_size : int
        batch size used for both pretraining and finetuning
    pretraining_noise_type : str
        The type of noise to be used in the pretraining phase
    pretraining_noise_parameter : float
        The parameter of the noise used for the pretraining phase
    hidden_layers : List[int]
        The array where each entry represents the number of nodes for
        each encoder layer. The decoder is the mirror
    learning_rate_pretraining : float
        The learning rate used in the pretraining optimization
    learning_rate_finetuning : float
        The learning rate used in the finetuning optimization
    epoch_pretraining : int
        The number of epochs used for the pretraining phase
    epoch_finetuning : int
        The number of epochs used for the finetuning phase
    finetuning_noise_type : str
        The type of noise to be used in the pretraining phase (default is None)
    finetuning_noise_parameter : float
        The parameter of the noise used for the pretraining phase (default is None)
    loss : float
        The resulting loss (default is None)

    Returns
    -------
    str
        The name of the file to be saved

    """

    return f"BATCH_SIZE_{str(batch_size)}" \
           f"_PRETR_NOISE_TYPE_{pretraining_noise_type}" \
           f"_PRETR_NOISE_PAR_{str(pretraining_noise_parameter).replace('.', ',')}" \
           f"_FINET_NOISE_TYPE_{finetuning_noise_type}" \
           f"_FINET_NOISE_PAR_{str(finetuning_noise_parameter).replace('.', ',')}" \
           f"_HID_LAYERS_[{','.join([str(elem) for elem in hidden_layers])}]" \
           f"_PRETR_LR_{str(learning_rate_pretraining).replace('.', ',')}" \
           f"_FINET_LR_{str(learning_rate_finetuning).replace('.', ',')}" \
           f"_EPOCH_PRETR_{str(epoch_pretraining)}" \
           f"_EPOCH_FINET_{str(epoch_finetuning)}" \
           f"_LOSS_{str(loss).replace('.', ',')}"


def reconstruct(train_dl, final_model_early_stopping, optimal_batch_size, optimal_pretraining_noise_type,
                optimal_pretraining_noise_parameter, optimal_hidden_layers, optimal_pretraining_learning_rate,
                optimal_finetuning_learning_rate, epochs_pretraining, optimal_epoch, start_name,
                optimal_finetuning_noise_type=None, optimal_finetuning_noise_parameter=None,  # ONLY FOR DENOSING AUTOENCODERS
                PICTURE_DIMENSION=28, NUMBER_OF_PICTURES_TO_DISPLAY=10):
    """Reconstructs a set of random pictures after reducing the dimensions to the intrinsic dimensionality used and
    saves the reconstructions in a .png file

    Parameters
    ----------
    train_dl : DataLoader
        the DataLoader containing the pictures to be reconstructed
    final_model_early_stopping : nn.Model
        the autoencoder model to be used to reduce the dimensionality
    optimal_batch_size : int
        batch size used for both pretraining and finetuning
    optimal_pretraining_noise_type : str
        The type of noise to be used in the pretraining phase
    optimal_pretraining_noise_parameter : float
        The parameter of the noise used for the pretraining phase
    optimal_hidden_layers : List[int]
        The array where each entry represents the number of nodes for
        each encoder layer. The decoder is the mirror
    optimal_pretraining_learning_rate : float
        The learning rate used in the pretraining optimization
    optimal_finetuning_learning_rate : float
        The learning rate used in the finetuning optimization
    epochs_pretraining : int
        The number of epochs used for the pretraining phase
    optimal_epoch : int
        The number of epochs used for the finetuning phase
    start_name : str
        Name the user wants to give to the final file saved
    optimal_finetuning_noise_type : str
        The type of noise to be used in the pretraining phase (default is None)
    optimal_finetuning_noise_parameter : float
        The parameter of the noise used for the pretraining phase (default is None)
    PICTURE_DIMENSION : int
        Number of pixels per dimension for hte final picture (default is 28)
    NUMBER_OF_PICTURES_TO_DISPLAY : int
        Number of random the user wants to reconstruct (default is 10)

    Returns
    -------

    """
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
        plt.imshow(
            final_model_early_stopping(features[0]).detach().numpy().reshape(PICTURE_DIMENSION, PICTURE_DIMENSION))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == NUMBER_OF_PICTURES_TO_DISPLAY-1:
            break

    plt.savefig(start_name +
                create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                             optimal_hidden_layers, optimal_pretraining_learning_rate, optimal_finetuning_learning_rate,
                             epochs_pretraining, optimal_epoch,
                             optimal_finetuning_noise_type,
                             optimal_finetuning_noise_parameter))  # ONLY FOR DENOSING AUTOENCODERS
