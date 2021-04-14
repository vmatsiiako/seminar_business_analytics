import torch
import torch.nn as nn


class FinetuningAE(nn.Module):
    """
        A class used to represent the Deep Autoencoder used for finetuning constructed using a list of
        stacked denoising autoencoder models

        ...

        Attributes
        ----------
        models : array
            The list of pretrained stacked denoising autoencoders used to initialise the Deep Autoencoder

        Methods
        -------
        forward(v)
            Performs a forward step over the full autoencoder (encodes and decodes)

        encode(v)
            Encodes the input (v) into the lower dimension defined by the coding layer dimension

        decode(h)
            Reconstructs the encoded information (h) into the dimensions of the original observations


        """

    def __init__(self, models):
        """
        Parameters
        ----------
        models : array
            The list of pretrained stacked denoising autoencoders used to initialise the Deep Autoencoder
        """

        super(FinetuningAE, self).__init__()

        # extract weights from each model, extract only the encoder because the weights of the decoder part
        # are simply the transposes
        encoders = []
        encoder_biases = []
        decoder_biases = []
        for model in models:
            encoders.append(nn.Parameter(model.W_encoder.clone()))
            encoder_biases.append(nn.Parameter(model.h_bias.clone()))
            decoder_biases.append(nn.Parameter(model.h_bias.clone()))

        # Initialize the output bias at random because in the pretraining no bias of 784 dimensions in trained
        decoder_biases.pop()
        decoder_biases.insert(0, nn.Parameter(torch.randn(784) * 0.1))

        # Mirror the biases of the encoder
        decoder_biases = reversed(decoder_biases)

        # build encoders and decoders
        self.encoders = nn.ParameterList(encoders)
        self.encoder_biases = nn.ParameterList(encoder_biases)
        self.decoder_biases = nn.ParameterList(decoder_biases)

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

        # Loop over all the layers that form the encoder of this Deep Autoencoder
        for i in range(len(self.encoders)):
            activation = torch.mm(p_v, self.encoders[i]) + self.encoder_biases[i]

            # Apply sigmoid function for non-linearity
            p_v = torch.sigmoid(activation)
        # for the last layer, we want a linear activation and thus return the activation
        return activation

    def decode(self, h):
        """Decodes the encoded information (h) into the dimensions of the original observations

        Parameters
        ----------
        h : DataLoader
            The (batch of) observations to be decoded
        """

        p_h = h

        # Loop over all the layers that form the decoder of this Deep Autoencoder
        for i in range(len(self.encoders)):
            # For the decoder, use the transposes of the encoder weights
            activation = torch.mm(p_h, self.encoders[len(self.encoders) - 1 - i].t()) + self.decoder_biases[i]

            # Apply sigmoid function for non-linearity
            p_h = torch.sigmoid(activation)
        # for the last layer, we want a linear activation and thus return the activation
        return activation
