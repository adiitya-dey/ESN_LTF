import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.EchoStateNetwork import ESN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a long enough positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)  # Save as a buffer, not a parameter

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)  # Input layer to hidden layer
        # self.relu = nn.Tanh()  # Activation function
        self.fc2 = nn.Linear(hidden_size, out_features)  # Hidden layer to output layer
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


        self.input_seg = self.seq_len // self.window_len # n = L / w
        self.pred_seg = self.pred_len  // self.window_len # m = H / w

        self.individual = configs.individual
        self.channels = configs.enc_in
        
        # Per-channel ESN (no-mixing allowed)
        self.projector = nn.ModuleList()
        self.predictor = nn.ModuleList()
        self.projector2 = nn.ModuleList()
 

        self.pos = PositionalEncoding(d_model=4)

        for i in range(self.channels):

            self.projector.append(nn.Linear(in_features=1,
                                            out_features=4,
                                            bias=False))
            
            self.predictor.append(FFN(in_features=self.seq_len, 
                                        out_features=self.pred_len, 
                                        hidden_size=32))
            
            self.projector2.append(nn.Linear(in_features=4,
                                            out_features=1,
                                            bias=False))
            
          
        

        
        
        
            
            
        
            
    ## x: [Batch, Input length, Channel]
    def forward(self, x):
        batch, _, _ = x.shape

        # Normalize
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - seq_mean
        
        
    

        
        y = torch.zeros(x.shape[0], self.channels, self.pred_len)
        for i in range(self.channels):

            # x: [B, L] -> [B, L, 1]
            seg = self.projector[i](x[:,:, i].unsqueeze(-1))

            seg = self.pos(seg)
            
            seg = self.predictor[i](seg.permute(0,2,1)) 

            seg = self.projector2[i](seg.permute(0,2,1))
          
            y[:,i,:] = seg.squeeze(-1)


        # De-Normalize
        y = y.permute(0,2,1) + seq_mean
      

        return y