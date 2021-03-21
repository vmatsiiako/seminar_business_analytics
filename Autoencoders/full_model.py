import torch
import torch.nn as nn

class full_model(nn.Model):
    def __init__(self, visible_dim, hidden_dim):

    class d_DAE(nn.Module):
        def __init__(self, visible_dim, hidden_dim):
            super(d_DAE, self).__init__()
            self.visible_dim = visible_dim
            self.hidden_dim = hidden_dim
            self.W_encoder = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.1)
            self.W_decoder = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.1)
            self.h_bias = nn.Parameter(torch.zeros(hidden_dim))  # v --> h
            self.v_bias = nn.Parameter(torch.zeros(visible_dim))  # h --> v

        def forward(self, v):
            """Forward step"""
            p_h = self.encode(v)
            return self.decode(p_h)

        def encode(self, v):
            """Encode input"""
            p_v = v
            activation = v
            activation = torch.mm(p_v, self.W_encoder) + self.h_bias
            p_v = torch.sigmoid(activation)
            # for the last layer, we want to return the activation directly rather than the sigmoid
            return activation

        def decode(self, h):
            """Decode hidden layer"""
            p_h = h
            activation = torch.mm(p_h, self.W_decoder.t()) + self.v_bias
            return activation

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

    def fit(self):
        # use the training here
        return None