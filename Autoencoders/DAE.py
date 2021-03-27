import torch
import torch.nn as nn

class DAE(nn.Module):
    def __init__(self, models):
        """Create a deep autoencoder based on a list of RBM models"""
        super(DAE, self).__init__()

        # extract weights from each model
        encoders = []
        encoder_biases = []
        decoders = []
        decoder_biases = []
        for model in models:
            encoders.append(nn.Parameter(model.W_encoder.clone()))
            encoder_biases.append(nn.Parameter(model.h_bias.clone()))
            decoders.append(nn.Parameter(model.W_encoder.clone()))
            decoder_biases.append(nn.Parameter(model.h_bias.clone()))

        decoder_biases.pop()
        decoder_biases.insert(0, nn.Parameter(torch.randn(784) * 0.1))
        decoder_biases = reversed(decoder_biases)

        # build encoders and decoders
        self.encoders = nn.ParameterList(encoders)
        self.encoder_biases = nn.ParameterList(encoder_biases)
        self.decoders = nn.ParameterList(reversed(decoders))
        self.decoder_biases = nn.ParameterList(decoder_biases)

        # # build encoders and decoders based on weights from each
        # encoders = nn.ParameterList([nn.Parameter(model.W.clone()) for model in models])
        # encoder_biases = nn.ParameterList([nn.Parameter(model.h_bias.clone()) for model in models])
        # decoders = nn.ParameterList([nn.Parameter(model.W.clone()) for model in reversed(models)])
        # decoder_biases = nn.ParameterList([nn.Parameter(model.v_bias.clone()) for model in reversed(models)])

    def forward(self, v):
        """Forward step"""
        p_h = self.encode(v)
        return self.decode(p_h)

    def encode(self, v):
        """Encode input"""
        p_v = v
        activation = v
        for i in range(len(self.encoders)):
            activation = torch.mm(p_v, self.encoders[i]) + self.encoder_biases[i]
            p_v = torch.sigmoid(activation)
        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    def decode(self, h):
        """Decode hidden layer"""
        p_h = h
        for i in range(len(self.encoders)):
            activation = torch.mm(p_h, self.decoders[i].t()) + self.decoder_biases[i]
            p_h = torch.sigmoid(activation)
        return activation