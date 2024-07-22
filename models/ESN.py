from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PrimaryNetwork(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super(PrimaryNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Linear(in_features=self.in_features,
                            out_features=self.out_features,
                            bias=True)
        
    def forward(self, x, weights, bias):
            self.fc.weight = nn.Parameter(weights)
            self.fc.bias = nn.Parameter(bias)
            return self.fc(x)
    
class HyperNetwork(nn.Module):

    def __init__(self, h_size, primary_input_size, primary_output_size) -> None:
        super(HyperNetwork, self).__init__()
        self.primary_input_size = primary_input_size
        self.primary_output_size = primary_output_size
        self.weight_gen = nn.Linear(h_size, self.primary_input_size * self.primary_output_size)
        self.bias_gen = nn.Linear(h_size, self.primary_output_size)

    def forward(self, h):
        weights = self.weight_gen(h)
        bias = self.bias_gen(h)
        
        ## Reshape weights to match primary network's weight shape
        weights = weights.view(-1, self.primary_output_size, self.primary_input_size)
        return weights, bias
    



class ESN(nn.Module):

    def __init__(self, 
                 input_size: int,
                 reservoir_size = 50,
                 activation = nn.Tanh(),
                 *args, **kwargs) -> None:
        super(ESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        
        self.leaking_rate = nn.Parameter(torch.tensor([0.5]), requires_grad=True)


        ## Initial state as a Zero Vector.
        # self.state = torch.zeros(self.reservoir_size)

        ## Non-Trainable Input projections will perform B u(t).
        self.input_projection = nn.Linear(in_features=self.input_size,
                                    out_features=self.reservoir_size,
                                    bias=False)
        
        with torch.no_grad():
            w_b = torch.empty(self.reservoir_size, self.input_size)
            self.input_projection.weight = nn.Parameter(nn.init.uniform_(w_b, a=-1.0, b=1.0), requires_grad=False)
        self.input_projection.weight.requires_grad_ = False


        ## Non Trainable State projections will perform A x(t).
        ##Initialize Reservoir Weight matrix.
        self.state_projection = nn.Linear(in_features=self.reservoir_size,
                                          out_features=self.reservoir_size,
                                          bias = False)
        with torch.no_grad():
            w_d =  torch.empty(self.reservoir_size)
            d = nn.init.uniform_(w_d, a=-0.5, b=0.5)
            self.state_projection.weight = nn.Parameter(torch.diag(d).to_sparse(), requires_grad=False)
        self.state_projection.weight.requires_grad_ = False



                     
    ## Calculate state.
    def get_state(self, input, state):
        ## Input = [Batch, 1, reservoir size]
        new_state = self.state_projection(state)
        new_state = self.activation(new_state + input)
        new_state = torch.mul(self.leaking_rate, state) + torch.mul((1 - self.leaking_rate), new_state)
        return new_state

    def forward(self, x):
        ## Input X: [Batch, Segment, Window]
         
        ## Output X: [Batch, segment, reservoir size]
        x = self.input_projection(x)

        ## All states = [Batch, Input Segment +1, reservoir size, 1]
        all_states = torch.zeros(x.shape[0], x.shape[1], self.reservoir_size)

        ## Iterate through Input sequence wise.
        for t in range(x.shape[1]):
            all_states[:,t,:] = self.get_state(x[:,t,:], all_states[:, t, :].clone())
        
        return all_states
       


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

        # Decompsition Kernel Size
        # kernel_size = 25
        # self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in


        
        self.esn_layer = ESN(reservoir_size=self.reservoir_size,
                                            activation=nn.Tanh(),
                                            input_size=self.window_len,
                                            )
        

        self.output_layer1 = nn.Linear(in_features=(self.input_seg - self.washout)*self.reservoir_size,
                                                 out_features=self.pred_seg*self.reservoir_size,
                                                 bias=True)
        # self.dropout = nn.Dropout(0.2)

        self.projection = nn.Linear(in_features=self.reservoir_size,
                                    out_features=self.window_len,
                                    bias=True)
            
       
    def forward(self, x):
        ## x: [Batch, Input length, Channel]
        ## batch, seq_len, channel = x.shape

        ## Output x: [Batch, Input Segment, window length]
        x = x.squeeze(-1)
        x = x.reshape(x.shape[0], self.input_seg, self.window_len)

        ## Output x: [Batch, Input Segment, reservoir size]
        x = self.esn_layer(x)

        ## Washout
        ## Output x: [Batch, (Input Segment - Washout), reservoir size]
        x = x[:, self.washout:, :]
        

        ## Flattening of batch sequences.
        ## Output x: [Batch, (Input Segment - Washout) * reservoir size)]
        x = x.view(x.shape[0], -1)

        ## Trainable Prediction layer.
        ## Output x: [Batch, Pred Segment, reservoir size]
        x = self.output_layer1(x)

        
        x = x.reshape(x.shape[0], self.pred_seg, self.reservoir_size)

        ## Trainable Projection Layer.
        ## output x: [Batch, Pred Length]
        # x= self.dropout(x)
        x = self.projection(x)
        x = x.reshape(x.shape[0], -1)

        # Add Channel to dimension.
        x = torch.unsqueeze(x, 1)

        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
    # def reset(self):
    #     self.esn_layer.reset_states()