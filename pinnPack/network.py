import torch
import torch.nn as nn

class physicsNetwork(nn.Module):
    '''
    Build PINN model for the heat equation with input shape (t,x)
    output shape u(t,x)
    Inputs:
      num_inputs: number of input variables
      layers: hidden layers
      activation: hidden layers activation function
      num_outputs: number of output variables
    Outputs:
      keras model
    '''
    
    def __init__(self, num_inputs = 2, layers = [32,32,32,32,32],
            activation = 'tanh', num_outputs = 2, seed = 42):
        super(physicsNetwork, self).__init__()
        torch.manual_seed(seed) 
        
        act_fn = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU()
        }.get(activation.lower(), nn.Tanh())
        
        #Building Network 
        layer_dims = [num_inputs] + layers + [num_outputs] 
        self.layers = nn.ModuleList() 
        self.activation = act_fn
        
        for i in range(len(layer_dims)-1): 
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x)) 
        return self.layers[-1](x)
    