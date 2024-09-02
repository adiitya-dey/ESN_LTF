import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from scipy.fft import dct, idct


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

class DFT(Function):
    @staticmethod
    def forward(ctx, input):
        transformed = torch.fft.rfft(input, dim=-1, norm="ortho")
        output = torch.stack((transformed.real, transformed.imag), dim=-1).to(input.device)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        real = grad_output[:,:,:,0]
        imag = grad_output[:,:,:,1]
        complex_tensor = torch.complex(real, imag)
        inverse_transformed = torch.fft.irfft(complex_tensor, dim = -1, norm="ortho").to(grad_output.device)
        return inverse_transformed
    

class IDFT(Function):
    @staticmethod
    def forward(ctx, input):
        real = input[:,:,:,0]
        imag = input[:,:,:,1]
        complex_tensor = torch.complex(real, imag)
        transformed = torch.fft.irfft(complex_tensor, dim=-1, norm="ortho").to(input.device)
        return transformed
    
    @staticmethod
    def backward(ctx, grad_output):
        transformed = torch.fft.rfft(grad_output, dim=-1, norm="ortho")
        output = torch.stack((transformed.real, transformed.imag), dim=-1).to(grad_output.device)
        return output

# class StateSpace(nn.Module):
#     def __init__(self):
#         self.A = torch.rand(1)


#     # x : [batch, features, seq_length]
#     def forward(self, x):
#         state = torch.zeros(x.shape)
#         for 



class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.channels = configs.enc_in
        self.window_len = 12
        self.input_seg = self.seq_len // self.window_len
        self.pred_seg = self.pred_len // self.window_len

        self.seg_predictor = nn.ModuleList()
        self.full_predictor = nn.ModuleList()

        self.amp_pred = nn.ModuleList()
        self.phase_pred = nn.ModuleList()

        for i in range(self.channels):
            self.seg_predictor.append(nn.Linear(self.input_seg, self.pred_seg))
            self.full_predictor.append(nn.Linear(self.seq_len, self.pred_len))


            self.amp_pred.append(nn.Linear(self.input_seg, self.pred_seg))
            self.phase_pred.append(nn.Linear(self.input_seg, self.pred_seg))
    


    def forward(self, x):
        # x: [Batch, Input length, Channel]

        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - seq_mean

        x = x.permute(0,2,1)
        

        out = torch.zeros(x.shape[0], self.channels, self.pred_len)
        for i in range(self.channels):
            # seg = x[:,i,:].reshape(-1, self.input_seg, self.window_len)
            full = x[:, i, :].reshape(-1, self.input_seg, self.window_len)

            # seg = DCT.apply(seg)
            full = DFT.apply(full)

            amp = full[:,:,:,0]
            phase = full[:,:,:,1]

            


            # y_seg= self.seg_predictor[i](seg.permute(0,2,1))
            y_amp = self.amp_pred[i](amp.permute(0,2,1))
            y_phase = self.phase_pred[i](phase.permute(0,2,1))

            # y_seg = IDCT.apply(y_seg.permute(0,2,1))
            y = IDFT.apply(torch.stack((y_amp.permute(0,2,1), y_phase.permute(0,2,1)), dim=-1))

            # y_seg = y_seg.reshape(x.shape[0], -1)

            y = y.reshape(x.shape[0], -1) # y_seg + 

            out[:,i,:] = y






        
        return out.permute(0,2,1) + seq_mean # [Batch, Output length, Channel]


    # def forward(self, x):
    #     # x: [Batch, Input length, Channel]

    #     seq_mean = torch.mean(x, dim=1, keepdim=True)
    #     x = x - seq_mean

    #     x = x.permute(0,2,1)
        

    #     out = torch.zeros(x.shape[0], self.channels, self.pred_len)
    #     for i in range(self.channels):
    #         # seg = x[:,i,:].reshape(-1, self.input_seg, self.window_len)
    #         full = x[:, i, :].unsqueeze(1)

    #         # seg = DCT.apply(seg)
    #         full = DCT.apply(full)

            


    #         # y_seg= self.seg_predictor[i](seg.permute(0,2,1))
    #         y_full = self.full_predictor[i](full)

    #         # y_seg = IDCT.apply(y_seg.permute(0,2,1))
    #         y_full = IDCT.apply(y_full).squeeze(1)

    #         # y_seg = y_seg.reshape(x.shape[0], -1)

    #         y = y_full # y_seg + 

    #         out[:,i,:] = y






        
    #     return out.permute(0,2,1) + seq_mean # [Batch, Output length, Channel]
    


    