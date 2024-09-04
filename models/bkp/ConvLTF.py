import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)  # Input layer to hidden layer
        # self.relu = nn.Tanh()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)  # Hidden layer to output layer
        # self.dropout = nn.Dropout(0.2) 

    
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
        


        self.input_seg = self.seq_len // self.window_len
        self.input_pad = self.seq_len % self.window_len

        self.pred_seg = self.pred_len // self.window_len

        self.individual = configs.individual
        self.channels = configs.enc_in

        self.conv_1x1 = nn.Conv1d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  bias=True,
                                  kernel_size=1,
                                  stride=1,
                                  
                                  )
        
        self.down_sampling = nn.Conv1d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  bias=False,
                                  kernel_size=self.window_len,
                                  stride=self.window_len)
        
        

        self.up_sampling = nn.ConvTranspose1d(in_channels=self.channels,
                                  out_channels=self.channels,
                                  bias=False,
                                  kernel_size=self.window_len,
                                  stride=self.window_len)
        
        
        self.ffn = FFN(input_size=self.input_seg,
                       output_size=self.pred_seg,
                       hidden_size=32)
        
      
        

        

        
            
    ## Input x: [Batch, Length, Channels]
    def forward(self, x):

        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = x - seq_mean
        
        # x = x.permute(0, 2, 1)
        
        # Input x: [B, L, C]
        # Output x: [B, C, L]
        x = x.permute(0, 2, 1) 

        # Input x: [B, C, L]
        # Output x: [B, C, L]
        x = self.conv_1x1(x)

        x = F.tanh(x)

        # Need to perform padding.
        if self.input_pad != 0:
            
            if self.input_pad%2 !=0:

                front = x[:, :, 0:1].repeat(1, 1, self.input_pad//2)
                end = x[:, :, -1:].repeat(1, 1, self.input_pad//2 + 1)
                x = torch.cat([front, x, end], dim=2)
            else:
                front = x[:, :, 0:1].repeat(1, 1, self.input_pad//2)
                end = x[:, :, -1:].repeat(1, 1, self.input_pad//2)
                x = torch.cat([front, x, end], dim=2)
        
        # Input x: [B, C, L]
        # Output x: [B, C, N] where N = L / w
        x = self.down_sampling(x)

        # x = F.sigmoid(x)

        # Input x: [B, C, N]
        # Output X: [B, C, M] where M = H / w
        x = self.ffn(x)

        # x = F.tanh(x)

        # x = self.layer_norm(x)

        # Input x: [B, C, M]
        # Output X: [B, C, H] 
        x = self.up_sampling(x)

        # x = x - in_c_mix.permute(0,2,1)
        # out_c_mix = self.out_channel_mixing(x.permute(0,2,1))

        # x = x + out_c_mix.permute(0,2,1)

        x = self.conv_1x1(x)

        x = x.permute(0, 2, 1) + seq_mean 
               

        return x