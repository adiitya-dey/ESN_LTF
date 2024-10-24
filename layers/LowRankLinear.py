import torch
import torch.nn as nn
from torch import autograd


class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # Save input and weight for backward pass
        ctx.save_for_backward(input, weight, bias)
        output = torch.matmul(input, weight)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, weight, bias = ctx.saved_tensors

        # Compute gradients for input, weight, and bias
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output.squeeze(1), weight.T)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.T, input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        w = torch.empty(in_features, out_features)
        w = nn.init.xavier_uniform_(w)
        self.W = nn.Parameter(w)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None

    def forward(self, x):
       
        out = CustomLinearFunction.apply(x, self.W, self.bias)
        return out



class ReducedRankFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, U, S, VT, bias=None):
        # Save input and weight for backward pass
        ctx.save_for_backward( input, U, S, VT, bias)
        output = torch.matmul(torch.matmul(torch.matmul(input, U), S), VT)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, U, S, VT, bias = ctx.saved_tensors

        # Compute gradients for input, weight, and bias
        grad_input = grad_U = grad_S = grad_VT = grad_bias = None

        if ctx.needs_input_grad[0]:  # grad_output shape: (b, c, H)
            # Calculate gradient with respect to the input
            grad_input = torch.matmul(grad_output.squeeze(1), torch.matmul(torch.matmul(U, S), VT).T)

        if ctx.needs_input_grad[1]:  # grad_U
            # Calculate gradient with respect to U
            grad_U = torch.matmul(input.permute(0,2,1), torch.matmul(grad_output, torch.matmul(S, VT).T)).squeeze(0)

        if ctx.needs_input_grad[2]:  # grad_S
            # Calculate gradient with respect to S
            grad_S = torch.matmul(torch.matmul(input,U).T, torch.matmul(grad_output, VT.T))

        if ctx.needs_input_grad[3]:  # grad_VT
            # Calculate gradient with respect to VT
            grad_VT = torch.matmul(torch.matmul(input, torch.matmul(U,S)).T, grad_output)

        if bias is not None and ctx.needs_input_grad[4]:  # grad_bias
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_U, grad_S, grad_VT, grad_bias


class ReducedRankLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ReducedRankLinear, self).__init__()
        rank = 15
        wU = torch.empty(in_features, rank)
        wS= torch.empty(rank, rank)
        wVT = torch.empty(rank, out_features)

        self.U = nn.Parameter(nn.init.xavier_uniform_(wU))
        self.S = nn.Parameter(nn.init.xavier_uniform_(wS))
        self.VT = nn.Parameter(nn.init.xavier_uniform_(wVT))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None

    def forward(self, x):
       
        out = ReducedRankFunction.apply(x, self.U, self.S, self.VT, self.bias)
        return out