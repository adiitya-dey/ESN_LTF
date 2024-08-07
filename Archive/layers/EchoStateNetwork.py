import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

   
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
        
        # Initalize Leaking reate as trainable parameter.
        # self.leaking_rate = nn.Parameter(torch.tensor([0.5]), requires_grad=True)


        ## Non-Trainable Input projections will perform B . u(t).

        ## Intialize Input matrix B as uniform matrix of [-1,1].
        self.input_projection = nn.Linear(in_features=self.input_size,
                                    out_features=self.reservoir_size,
                                    bias=False)
        
        with torch.no_grad():
            w_b = torch.empty(self.reservoir_size, self.input_size)
            self.input_projection.weight = nn.Parameter(nn.init.uniform_(w_b, a=-1.0, b=1.0), requires_grad=False)
        self.input_projection.weight.requires_grad_ = False


        ## Non Trainable State projections will perform A . x(t).

        ##Initialize Reservoir Weight matrix A as a Diagonal matrix with diagonal values between [-.5, .5].
        self.state_projection = nn.Linear(in_features=self.reservoir_size,
                                          out_features=self.reservoir_size,
                                          bias = False)
        with torch.no_grad():
            w_d =  torch.empty(self.reservoir_size)
            d = nn.init.uniform_(w_d, a=-0.5, b=0.5)
            self.state_projection.weight = nn.Parameter(torch.diag(d).to_sparse(), requires_grad=False)
        self.state_projection.weight.requires_grad_ = False



                     
    ## Calculate state.
    ## input : [batch, 1, reservoir size]
    def get_state(self, input, state):
        
        new_state = self.state_projection(state)
        new_state = self.activation(new_state + input)
        similarity = nn.CosineSimilarity(dim=1)(new_state, state).unsqueeze(1)
        leaking_rate = F.tanh(similarity)
        # new_state = self.leaking_rate * state +(1 - self.leaking_rate) * new_state
        new_state = 0.0 * state +(1 - 0.0) * new_state
        return new_state


    ## x: [Batch, Segment, Window]
    def forward(self, x):
        
         
        ## x: [Batch, segment, reservoir size]
        x = self.input_projection(x)

        ## all_states : [Batch, Input Segment +1, reservoir size, 1]
        all_states = torch.zeros(x.shape[0], x.shape[1] + 1, self.reservoir_size)

        ## Iterate through Input sequence wise.
        for t in range(1, x.shape[1]):
            all_states[:,t,:] = self.get_state(x[:,t,:], all_states[:, t -1, :].clone())
        
        ## all_states : [Batch, Input Segment +1, reservoir size, 1]
        return all_states[:,1:,:]