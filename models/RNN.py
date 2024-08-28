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
        # self.esn_modules = nn.ModuleList()
        self.seg_predictor = nn.ModuleList()
        self.state_projector = nn.ModuleList()
        self.segmentor = nn.ParameterList()
        # self.reverse_segmentor = nn.ModuleList()

        for i in range(self.channels):
            self.segmentor.append(nn.Parameter(torch.rand(self.window_len, 1), requires_grad=True))
            
            # self.esn_modules.append(ESN(reservoir_size=self.reservoir_size,
            #                             input_size=1,
            #                             activation=nn.Tanh(),
            #                             ))
            
            # self.seg_predictor.append(nn.Linear(in_features=self.input_seg, 
            #                                     out_features=self.pred_seg, 
            #                                     bias=True))

            # Replace Linear seg_predictor with LSTM
            self.seg_predictor.append(nn.LSTM(input_size=1, 
                                              hidden_size=self.reservoir_size, 
                                              num_layers=1, 
                                              batch_first=True))
            
            self.state_projector.append(nn.Linear(in_features=self.reservoir_size, 
                                             out_features=1, 
                                             bias=True))
            
            # self.reverse_segmentor.append(nn.Linear(in_features=1, 
            #                                         out_features=self.window_len, 
            #                                         bias=False))

            # with torch.no_grad():
            #     self.reverse_segmentor.weight = nn.Parameter(self.segmentor[i].weight.t())
        

        
        
        
            
            
        
            
    ## x: [Batch, Input length, Channel]
    def forward(self, x):
        batch, _, _ = x.shape

        # Normalize
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - seq_mean
        

        
        # x: [B, L, c] -> [B, c, L]
        x = x.permute(0, 2, 1) 

        
        y = torch.zeros(x.shape[0], self.channels, self.pred_len)
        for i in range(self.channels):

            # x: [B, L] -> [B, n, w] -> [B, n, 1]
            # seg = self.segmentor[i](x[:,i,:].reshape(-1, self.input_seg, self.window_len))
            seg = torch.matmul (x[:,i,:].reshape(-1, self.input_seg, self.window_len),  self.segmentor[i])

            # x: [B, n, 1]-> [B, n, r]
            # seg = self.esn_modules[i](seg) 

            # Initialize hidden states for LSTM
            h_t = torch.zeros(1, batch, self.reservoir_size)
            c_t = torch.zeros(1, batch, self.reservoir_size)
            
            predictions = []
            input_t = seg

            for t in range(self.pred_seg):
                out, (h_t, c_t) = self.seg_predictor[i](input_t, (h_t, c_t))
                
                # Map to the desired output
                output = self.state_projector[i](out[:, -1, :])
                
                predictions.append(output)

                # Prepare the next input
                input_t = output.unsqueeze(1)

            # Concatenate all predictions
            seg = torch.unsqueeze(torch.cat(predictions, dim=1), -1)

            # x: [B, r, m] -> [B, m, r] -> [B, m, 1]
            # seg = self.state_projector[i](seg.permute(0,2,1))

            # x: [B, m, 1] -> [B, m , w]
            # seg = self.reverse_segmentor[i](seg)
            seg = torch.matmul(seg, self.segmentor[i].t())

            # x: [B, m, w] -> [B, H]
            y[:,i,:] = seg.reshape(batch, -1)


        # De-Normalize
        y = y.permute(0,2,1) + seq_mean
      

        return y