import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

   
class ESN(nn.Module):

    def __init__(self, 
                 input_size: int,
                 reservoir_size = 50,
                 connectivity_rate= 1.0,
                 spectral_radius=1.0,
                 activation = nn.Tanh(),
                 *args, **kwargs) -> None:

        super(ESN, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.activation = activation
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        
        # Initalize Leaking reate as trainable parameter.
        # self.leaking_rate = nn.Parameter(torch.tensor([0.5]), requires_grad=True)


        ## Non-Trainable Input projections will perform B . u(t).

        ## Intialize Input matrix B as uniform matrix of [-1,1].
        self.input_projection = nn.Linear(in_features=self.input_size,
                                    out_features=self.reservoir_size,
                                    bias=False)
        
        with torch.no_grad():
            w_b = torch.empty(self.reservoir_size, self.input_size)
            self.input_projection.weight = nn.Parameter(nn.init.uniform_(w_b, a=0.0, b=1.0), requires_grad=False)
        self.input_projection.weight.requires_grad_ = False


        ## Non Trainable State projections will perform A . x(t).

        ##Initialize Reservoir Weight matrix A as a Diagonal matrix with diagonal values between [-.5, .5].
        self.state_projection = nn.Linear(in_features=self.reservoir_size,
                                          out_features=self.reservoir_size,
                                          bias = False)
        with torch.no_grad():
        #     w_d =  torch.empty(self.reservoir_size)
        #     d = nn.init.uniform_(w_d, a=-1., b=1.)
        #     self.state_projection.weight = nn.Parameter(torch.diag(d).to_sparse(), requires_grad=False)
            # self.state_projection.weight = nn.Parameter(self.simple_reservoir_matrix(size=self.reservoir_size,connectivity_rate=1.0, spectral_radius=0.2).to_sparse(), requires_grad = False)
            self.state_projection.weight = nn.Parameter(self.create_beta_matrix(N=self.reservoir_size, alpha=.7).to_sparse(), requires_grad = False)
        self.state_projection.weight.requires_grad_ = False

  
                     
    ## Calculate state.
    ## input : [batch, 1, reservoir size]
    def get_state(self, input, state):
        
        new_state = self.state_projection(state)
        new_state = self.activation(new_state + input)
        # similarity = nn.CosineSimilarity(dim=1)(new_state, state).unsqueeze(1)
        # leaking_rate = F.tanh(similarity)
        leaking_rate = 0.0
        # new_state = self.leaking_rate * state +(1 - self.leaking_rate) * new_state
        new_state = leaking_rate * state +(1 - leaking_rate) * new_state
        return new_state


    ## x: [Batch, Segment]
    def forward(self, x):
        
        # x = torch.unsqueeze(x, -1)
        ## x: [Batch, segment, reservoir size]
        x = self.input_projection(x)

        ## all_states : [Batch, Input Segment +1, reservoir size, 1]
        all_states = torch.zeros(x.shape[0], x.shape[1] + 1, self.reservoir_size)

        ## Iterate through Input sequence wise.
        for t in range(1, x.shape[1]):
            all_states[:,t,:] = self.get_state(x[:,t,:], all_states[:, t -1, :].clone())
        
        ## all_states : [Batch, Input Segment +1, reservoir size, 1]
        return all_states[:,1:,:]
    

    ## Original Method to create Reserovir Matrix
    @staticmethod
    def simple_reservoir_matrix(size, connectivity_rate, spectral_radius):
        ## Initializing Reservoir Weights according to "Re-visiting the echo state property"(2012)
        ##
        ## Initialize a random matrix and induce sparsity.
        W_res = torch.rand((size, size))
        W_res[torch.rand(*W_res.data.shape) > connectivity_rate] = 0

        ## Scale the matrix based on user defined spectral radius.
        current_spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res)))
        W_res.data = W_res * (spectral_radius / current_spectral_radius)

        ## Induce half of the weights as negative weights.
        total_entries = size * size
        num_negative_entries = total_entries//2
        negative_indices = np.random.choice(total_entries, num_negative_entries, replace=False)
        W_flat = W_res.flatten()
        W_flat[negative_indices] *= -1
        W_res = W_flat.reshape(*W_res.shape)

        return W_res  
    
    @staticmethod
    def create_beta_matrix(N: int, alpha: float):
    # Define beta_+ and beta_-
        beta_plus = 0.5 * (1 - alpha)
        beta_minus = 0.5 * (1 + alpha)
        
        # Initialize the NxN matrix A with zeros
        A = torch.zeros((N, N))
        
        # Fill in the non-zero entries
        for i in range(N):
            A[i, (i + 1) % N] = beta_plus  # a_{i, i+1} (cyclic for last row)
            A[i, (i - 1) % N] = beta_minus  # a_{i, i-1} (cyclic for first row)
        
        return A
    

    @staticmethod
    def create_alpha_matrix(N, alpha):
        # Define beta_+ and beta_-
        beta_plus = alpha
        
        # Initialize the NxN matrix A with zeros
        A = torch.zeros((N, N))
        
        # Fill in the non-zero entries
        for i in range(N):
            A[i, (i + 1) % N] = beta_plus  # a_{i, i+1} (cyclic for last row)

        A = A + (1- alpha) *torch.eye(N)
        
        return A

