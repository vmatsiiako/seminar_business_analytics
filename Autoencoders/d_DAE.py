import torch
import torch.nn as nn


class d_DAE(nn.Module):
    def __init__(self,  visible_dim, hidden_dim):
        super(d_DAE,self).__init__()
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