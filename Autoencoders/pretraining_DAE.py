import torch
import torch.nn as nn


class PretrainingDAE(nn.Module):
    """
        A class used to represent the Single Layer Denoising Autoencoder used for pretraining

        ...

        Attributes
        ----------
        visible_dim : int
            The input dimension, that is the number of input nodes of the autoencoder
        hidden_dim: int
            The dimensions of the coding layer, that is the number of nodes of the hidden layer

        Methods
        -------
        forward(v)
            Performs a forward step over the autoencoder (encodes and decodes)

        encode(v)
            Encodes the input (v) into the lower dimension defined by the coding layer dimension

        decode(h)
            Reconstructs the encoded information (h) into the dimensions of the original observations


        """

    def __init__(self,  visible_dim, hidden_dim):
        """
        Parameters
        ----------
        visible_dim : int
            The input dimension, that is the number of input nodes of the autoencoder
        hidden_dim: int
            The dimensions of the coding layer, that is the number of nodes of the hidden layer
        """

        super(PretrainingDAE, self).__init__()

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        # Initialize at random the weights of the autoencoder
        self.W_encoder = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.1)
        self.W_decoder = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.1)

        # Initialize at zero the biases of the autoencoder
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))

    def forward(self, v):
        """Performs a forward step over the full autoencoder (encodes and decodes)

        Parameters
        ----------
        v : DataLoader
            The (batch of) observations to be passed through the autoencoder
        """

        p_h = self.encode(v)
        return self.decode(p_h)

    def encode(self, v):
        """Encodes the input (v) into the lower dimension defined by the coding layer dimension

        Parameters
        ----------
        v : DataLoader
            The (batch of) observations to be encoded
        """

        p_v = v
        activation = v
        activation = torch.mm(p_v, self.W_encoder) + self.h_bias
        p_v = torch.sigmoid(activation)
        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    def decode(self, h):
        """Decodes the encoded information (h) into the dimensions of the original observations

        Parameters
        ----------
        h : DataLoader
            The (batch of) observations to be decoded
        """

        p_h = h
        activation = torch.mm(p_h, self.W_decoder.t()) + self.v_bias
        return activation
