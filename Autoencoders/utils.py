import numpy as np
import matplotlib as plt


def add_noise(img, noise_type, percentage=None, sigma=None):
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        mean = 0
        noise = np.random.normal(mean, sigma, img.shape)
        img = img + noise
        return img

    if noise_type == "zeros":
        noise = np.random.choice([0, 1], size=(img.shape), p=[percentage, 1 - percentage])
        img = img * noise
        return img


def display_2d_repr(data, labels, fname=None):
    """Display a 2d representation of the MNIST digits
    Parameters
    ----------
    data: Tensor
        2d representation of MNIST digits
    labels: list
        the label for each data point in data
    fname: str
        filename to save plot in
    """

    digit_to_color = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                      "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    xs = np.array([x[0] for x in data])
    ys = np.array([x[1] for x in data])

    fig, ax = plt.subplots()
    labels_to_show = labels[0:len(data)]
    for digit in range(10):
        ix = np.where(labels_to_show == digit)
        ax.scatter(xs[ix], ys[ix], c=digit_to_color[digit],
                    label=digit, marker=".")
    ax.legend()
    if fname is not None:
        plt.savefig(fname)
    plt.show()