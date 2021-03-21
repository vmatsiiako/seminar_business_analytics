import torch
import torch.nn as nn

class full_model(nn.Module):

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

        def fit(self):
            for hidden_dim in HIDDEN_LAYERS:
                # train d_DAE
                dae = d_DAE(visible_dim=visible_dim, hidden_dim=hidden_dim)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(dae.parameters(), lr=0.01, weight_decay=1e-5)

                epochs = EPOCHS_PRETRAINING
                l = len(dae_train_dl_clean)
                losslist = list()
                epochloss = 0
                running_loss = 0
                dataset_previous_layer_batched = []
                for i, features in tqdm(enumerate(dae_train_dl_clean)):
                    dataset_previous_layer_batched.append(features[0])

                for epoch in range(epochs):

                    print("Entering Epoch: ", epoch)
                    for i, features in tqdm(enumerate(dae_train_dl_corrupted)):
                        # -----------------Forward Pass----------------------
                        output = dae(features[0])
                        loss = criterion(output, dataset_previous_layer_batched[i])
                        # -----------------Backward Pass---------------------
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        epochloss += loss.item()
                        # -----------------Log-------------------------------
                losslist.append(running_loss / l)
                running_loss = 0

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