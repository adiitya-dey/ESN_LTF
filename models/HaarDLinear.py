import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Function
from scipy.fft import dct, idct

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
    
class DCT(Function):
        @staticmethod
        def forward(ctx, input):
            # Convert PyTorch tensor to NumPy array
            input_np = input.cpu().numpy()
            # Apply DCT using scipy
            transformed_np = dct(input_np, type=2, norm="ortho", axis=-1, orthogonalize=True)
            # Convert back to PyTorch tensor
            output = torch.from_numpy(transformed_np).to(input.device)
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            # Convert gradient to NumPy array
            grad_output_np = grad_output.cpu().numpy()
            # Apply IDCT using scipy
            grad_input_np = idct(grad_output_np, type=2, norm='ortho', axis=-1, orthogonalize=True)
            # Convert back to PyTorch tensor
            grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
            return grad_input

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.low_pass_filter = torch.tensor([1, 1], dtype=torch.float32) / math.sqrt(2)

        self.low_pass_filter = self.low_pass_filter.reshape(1,1,-1).repeat(self.channels, 1, 1)


        if (self.seq_len%2)!=0:
            in_len = self.seq_len//2 + 1
        else:
            in_len = self.seq_len//2

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(in_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(in_len,self.pred_len))
                # self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                # self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(in_len,self.pred_len)
            self.Linear_Trend = nn.Linear(in_len,self.pred_len)
            # self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            # self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)


        ## Haar Decomp
        if (self.seq_len%2)!=0:
            seasonal_init = F.pad(seasonal_init, (0, 1))
            trend_init = F.pad(trend_init, (0, 1))


        seasonal_init = F.conv1d(input=seasonal_init, weight=self.low_pass_filter, stride=2, groups=self.channels)
        trend_init = F.conv1d(input=trend_init, weight=self.low_pass_filter, stride=2, groups=self.channels)

        #DCT
        seasonal_init = DCT.apply(seasonal_init) / seasonal_init.shape[-1]
        trend_init = DCT.apply(trend_init) / trend_init.shape[-1]


        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
    def step(self):
        pass
