import numpy as np
import matplotlib as plt


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

print(create_title(64, 'zeros', 0.5, [1, 2, 3, 4], 0.5, 0.5, 20, 50))

# def display_2d_repr(data, labels, fname=None):
#     """Display a 2d representation of the MNIST digits
#     Parameters
#     ----------
#     data: Tensor
#         2d representation of MNIST digits
#     labels: list
#         the label for each data point in data
#     fname: str
#         filename to save plot in
#     """
#
#     digit_to_color = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
#                       "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
#     xs = np.array([x[0] for x in data])
#     ys = np.array([x[1] for x in data])
#
#     fig, ax = plt.subplots()
#     labels_to_show = labels[0:len(data)]
#     for digit in range(10):
#         ix = np.where(labels_to_show == digit)
#         ax.scatter(xs[ix], ys[ix], c=digit_to_color[digit],
#                     label=digit, marker=".")
#     ax.legend()
#     if fname is not None:
#         plt.savefig(fname)
#     plt.show()