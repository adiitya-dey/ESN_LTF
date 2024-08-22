import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.EchoStateNetwork import ESN
   
class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        # self.relu = nn.Tanh()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer
        # self.dropout = nn.Dropout(0.2) 

    # x: [batch, input_seg, reservoir size]
    def forward(self, x):
        out = self.fc1(x)  # Linear transformation
        # out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Linear transformation
        # out = self.dropout(out)
        return out

class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.window_len = 12
        self.reservoir_size = 25
        self.washout = 10


        self.input_seg = self.seq_len // self.window_len
        self.pred_seg = self.pred_len // self.window_len

        self.individual = configs.individual
        self.channels = configs.enc_in

        
        self.esn = ESN(reservoir_size=self.reservoir_size,
                        activation=nn.Tanh(),
                        input_size=self.window_len,
                        )
        
        self.readout = FFN(input_size=self.input_seg - self.washout,
                            hidden_size=32,
                            output_size=self.pred_seg)

        # ## Readout Weights and Bias.
        # w_r = torch.empty(self.input_seg - self.washout, self.reservoir_size, self.pred_seg)
        # w_r = nn.init.constant_(w_r, 0.5)
        # self.readout_W = nn.Parameter(w_r, requires_grad=True)

        # w_b = torch.empty(self.pred_seg, self.reservoir_size)
        # w_b = nn.init.constant(w_b, 0.5)
        # self.readout_B = nn.Parameter(w_b, requires_grad=True)

        self.projection = nn.Linear(in_features=self.reservoir_size,
                                        out_features=self.window_len,
                                        bias=True)
            
    ## x: [Batch, Input length, Channel]
    def forward(self, x):
        
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.window_len, self.input_seg).permute(0, 2, 1)

        
        ## Output x: [bc, Input Segment, reservoir size]
        x = self.esn(x)

        ## Output x: [Batch, (Input Segment - Washout), reservoir size]
        x = x[:, self.washout:, :]
        
        x = x.permute(0,2,1)

        x = self.readout(x)

        x = x.permute(0, 2, 1)
        ## Trainable Prediction/Readout layer.
        ## Output x: [Batch, Pred Segment, reservoir size]
        ## b: batch, s: input_segment - washout, r: reservoir size, p: pred_segment
        # x = torch.einsum('bsr, srp -> bpr', x, self.readout_W) + self.readout_B

        ## Trainable Projection Layer.
        ## output x: [Batch, Pred Length]
        x = self.projection(x)
        x = x.reshape(batch_size, self.channels, self.pred_len)

         # permute and denorm
        x = x.permute(0, 2, 1) + seq_mean

        return x