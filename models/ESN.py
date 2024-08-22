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

        # Linear Projection
        self.conv_1x1 = nn.Conv1d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=1,
                                  stride=1)
        
        self.conv_wx1 = nn.Conv1d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=self.window_len,
                                  stride=1,
                                  padding="same")
        
        # Down-Sampling
        self.conv_wxw = nn.Conv1d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=self.window_len,
                                  stride=self.window_len)
        

        # self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.window_len // 2),
                                # stride=1, padding=self.window_len // 2, padding_mode="zeros", bias=False)
        
        # Per-channel ESN (no-mixing allowed)
        self.esn_modules = nn.ModuleList()
        self.out = nn.ModuleList()
        self.projection = nn.ModuleList()
        self.segmentor = nn.ModuleList()

        for i in range(self.channels):
            self.esn_modules.append(ESN(reservoir_size=self.reservoir_size,
                                        input_size=1,
                                        activation=nn.Tanh(),
                                        ))
            
            self.out.append(nn.Linear(in_features=self.input_seg, out_features=self.pred_seg, bias=False))
            self.projection.append(nn.Linear(in_features=self.reservoir_size, out_features=1, bias=False))
            self.segmentor.append(nn.Linear(in_features=self.window_len, out_features=1, bias=False))

        # Up-sampling
        # self.Tconv_wxw = nn.ConvTranspose1d(in_channels=self.channels,
        #                                     out_channels=self.channels,
        #                                     kernel_size=self.window_len,
                                            # stride=self.window_len)
        
        self.f_linear = nn.Linear(in_features=self.pred_seg,
                                  out_features=self.pred_len)
        
        
        
            
            
        
            
    ## x: [Batch, Input length, Channel]
    def forward(self, x):
        batch, _, _ = x.shape

        # Normalization
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = x - seq_mean
        
        # Input x: [B, L, C]
        # Output x: [B, C, L]
        x = x.permute(0, 2, 1) 

        ## Non-Linear projection
        # x = self.conv_1x1(x)
        

        # x = self.conv_wx1(x)
        # x = F.tanh(x)

        ## Downsampling
        # x = self.conv_wxw(x) # x: (B, C, N)

        # 1D convolution aggregation
        # x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.channels, self.seq_len) + x

        out = torch.zeros(x.shape[0], self.channels, self.pred_seg)
        for i in range(self.channels):
            segment = self.segmentor[i](x[:,i,:].reshape(-1, self.input_seg, self.window_len))
            states = self.esn_modules[i](segment.squeeze(-1)) # x: (B, N, r)
            states = self.out[i](states.permute(0, 2, 1)) # x: (B, M, r)
            out[:,i,:] = self.projection[i](states.permute(0,2,1)).squeeze(-1)

        y = out # x: (B ,C, M)

        # ## Upsampling
        # y = self.Tconv_wxw(y) #: (B, C, H)

        # x = self.conv_wx1(x)
        # y = self.conv_1x1(x)
        y = self.f_linear(y)


        y = y.permute(0,2,1) + seq_mean
      

        return y