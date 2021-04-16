import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def add_noise(img, noise_type, parameter=None):
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        mean = 0
        noise = np.random.normal(mean, parameter, img.shape)
        img = img + noise
        return img

    if noise_type == "zeros":
        noise = np.random.choice([0, 1], size=(img.shape), p=[parameter, 1 - parameter])
        img = img * noise
        return img

def create_title(batch_size, pretraining_noise_type, pretraining_noise_parameter, hidden_layers,
                 learning_rate_pretraining, learning_rate_finetuning, epoch_pretraining, epoch_finetuning,
                 finetuning_noise_type = None, finetuning_noise_parameter = None, loss = None):
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
                optimal_finetuning_noise_type, optimal_finetuning_noise_parameter,  # ONLY FOR DENOSING AUTOENCODERS
                PICTURE_DIMENSION=28, NUMBER_OF_PICTURES_TO_DISPLAY=10):
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
        if i == 9:
            break

    plt.savefig(start_name +
                create_title(optimal_batch_size, optimal_pretraining_noise_type, optimal_pretraining_noise_parameter,
                             optimal_hidden_layers, optimal_pretraining_learning_rate, optimal_finetuning_learning_rate,
                             epochs_pretraining, optimal_epoch,
                             optimal_finetuning_noise_type,
                             optimal_finetuning_noise_parameter))  # ONLY FOR DENOSING AUTOENCODERS
