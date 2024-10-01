import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from scipy.fft import dct, idct
import math

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


class IDCT(Function):
    @staticmethod
    def forward(ctx, input):
        # Convert PyTorch tensor to NumPy array
        input_np = input.cpu().numpy()
        # Apply IDCT using scipy
        transformed_np = idct(input_np, type=2, norm='ortho', axis=-1, orthogonalize=True)
        # Convert back to PyTorch tensor
        output = torch.from_numpy(transformed_np).to(input.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Convert gradient to NumPy array
        grad_output_np = grad_output.cpu().numpy()
        # Apply DCT using scipy
        grad_input_np = dct(grad_output_np, type=2, norm='ortho', axis=-1, orthogonalize=True)
        # Convert back to PyTorch tensor
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        return grad_input
    

class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer


    # x: [batch, input_seg, reservoir size]
    def forward(self, x):
        out = self.fc1(x)  # Linear transformation
        out = self.fc2(out)  # Linear transformation
        return out

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.channels = configs.enc_in

        self.window = 6
        
        low_pass_filter = torch.tensor([1, 1], dtype=torch.float32) / math.sqrt(2)
        high_pass_filter = torch.tensor([-1, 1], dtype=torch.float32) / math.sqrt(2)

        low_pass_filter = low_pass_filter.reshape(1,1,-1)
        high_pass_filter = high_pass_filter.reshape(1,1,-1)

        low_pass_filter = low_pass_filter.repeat(self.channels, 1, 1)
        high_pass_filter = high_pass_filter.repeat(self.channels, 1, 1)

        self.dec_lo = nn.Parameter(low_pass_filter, requires_grad=False)
        self.dec_hi = nn.Parameter(high_pass_filter, requires_grad=False)
        self.rec_lo = nn.Parameter(low_pass_filter, requires_grad=False)
        self.rec_hi = nn.Parameter(high_pass_filter, requires_grad=False)



        if (self.seq_len%2)!=0:
            in_len = self.seq_len//2 + 1
        else:
            in_len = self.seq_len//2


        if (self.pred_len%2) != 0:
            out_len = self.pred_len//2 + 1
        else:
            out_len = self.pred_len//2

        self.layer_lo = FFN(in_len,16,self.pred_len)
        self.layer_hi = FFN(in_len,16,self.pred_len)
        
        


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch, _, _ = x.shape

        ## Scaled Normalization
        x = x.permute(0,2,1)
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = (x - seq_mean)

        if (self.seq_len%2)!=0:
            x = F.pad(x, (0, 1))

        ## Haar decomposition
        s1 = F.conv1d(input=x, weight=self.dec_lo, stride=2, groups=self.channels)
        # d1 = F.conv1d(input=x, weight=self.dec_hi, stride=2, groups=self.channels)

        ## Cosine Transform
        s1 = DCT.apply(s1) / math.sqrt(s1.shape[-1])
        # d1 = DCT.apply(d1)

        # Cut-off Frequency
        # s1[:,:,s1.shape[-1]//3:] = 1e-12


        ## Prediction
        pred_s1 = self.layer_lo(s1)
        # pred_d1 = self.layer_hi(s1)

        ## Inverse Cosine Transform
        # pred_s1 = IDCT.apply(pred_s1)
        # pred_d1 = IDCT.apply(pred_d1)

        # ## Haar reconstruction
        # pred_s1 = F.conv_transpose1d(input=pred_s1, weight=self.rec_lo, stride=2, groups=self.channels)
        # pred_d1 = F.conv_transpose1d(input=pred_d1, weight=self.rec_hi, stride=2, groups=self.channels)

        # out = pred_s1 + pred_d1

        out = pred_s1 + seq_mean
        
        return out.permute(0,2,1) # [Batch, Output length, Channel]


# class DFT(Function):
#     @staticmethod
#     def forward(ctx, input):
#         transformed = torch.fft.rfft(input, dim=-1, norm="ortho").to(input.device)
#         return transformed
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         inverse_transformed = torch.fft.irfft(grad_output, dim = -1, norm="ortho").to(grad_output.device)
#         return inverse_transformed
    

# class IDFT(Function):
#     @staticmethod
#     def forward(ctx, input):
#         transformed = torch.fft.irfft(input, dim=-1, norm="ortho").to(input.device)
#         return transformed
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         transformed = torch.fft.rfft(grad_output, dim=-1, norm="ortho").to(grad_output.device)
#         return transformed
