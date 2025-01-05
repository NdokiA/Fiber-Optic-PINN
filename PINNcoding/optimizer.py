import tensorflow as tf
import tensorflow_probability as tfp
from gradient import nlseGradient

class optimizer():
    '''
    Algorithm for model backpropagation
    '''
    def __init__(self, model, batched_data,
                 alpha, beta2, gamma):
        self.model = model 
        
        self.col_data = batched_data['col_point']
        self.init_data = batched_data['init_point']
        self.bound_data = batched_data['bound_point']
        
        self.alpha = alpha 
        self.beta2 = beta2 
        self.gamma = gamma
        
        self.grads = nlseGradient(self.model)
        self.optimizer = tfp.optimizer.lbfgs_minimize
    
    def forward_prop(self, col_point):
        """
        Forward propagation of the model to find the residue and loss function

        Args:
            col_point (_type_): _description_
        """        
        u_eq, __, du_dx_eq, d2u_dt2_eq, __ = self.grads.call(self.col_data[0])
        u_init, __, __, __, __ = self.grads.call(self.init_data[0])
        u_bound, __, __, __, __ = self.grads.call(self.bound_data[0])
        
        u_eq, __, du_dx_eq, d2u_dt2_eq, __ = self.grads.call(self.col_data[0])
        u_init, __, __, __, __ = self.grads.call(self.init_data[0])
        u_bound, __, __, __, __ = self.grads.call(self.bound_data[0])
        residue = du_dx_eq + self.alpha/2*u_eq +1j/2*self.beta2*d2u_dt2_eq - self.gamma*u_eq**2*u_eq

