import torch
import torch.nn as nn

class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size):
        super(ESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        
        # Input-to-reservoir weights
        w_b = torch.empty(self.reservoir_size, self.input_size)
        self.W_in = nn.init.uniform_(w_b, a=-1.0, b=1.0)


        # Reservoir weights (with sparsity)
        w_d = torch.empty(self.reservoir_size)
        d = nn.init.uniform_(w_d, a=-1.0, b=1.0)
        self.W_res = torch.diag(d).to_sparse()
        
        # Initial state
        self.state = torch.zeros(1, reservoir_size)
        
    def forward(self, x):
        # x: (batch, channels, seq_len)
        batch_size, channels, seq_len = x.size()
        
        # Output container for states
        all_states = torch.zeros(batch_size, channels, seq_len, self.reservoir_size)
        
        for b in range(batch_size):
            for c in range(channels):
                state = self.state.clone().detach()  # Initialize the state for each channel
                
                for t in range(seq_len):
                    u = x[b, c, t].unsqueeze(0).unsqueeze(0)  # Shape (1, input_size)
                    A = torch.mm(u, self.W_in.T)
                    B = torch.mm(state, self.W_res)
                    state = torch.tanh(A + B)
                    all_states[b, c, t] = state
                
        return all_states  # Shape: (batch, channels, seq_len, reservoir_size)


