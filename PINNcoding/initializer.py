import tensorflow as tf 
import numpy as np

class initData:
    '''
    Initialize Dataset
    '''
    def __init__(self):
        self.tx_train = {}
        self.u_train = {}
        self.v_train = {}
    
    def add_collocation_point(self,tx):
        """
        Adding set of collocation points to self.tx_train, and
        expected residual value on self.u_train and self.v_train via tf.zeros_like

        Args:
            tx (array_like): Inirialized collocation point
        """        
        self.tx_train['col_point'] = tx
        self.u_train['col_point'] = tf.zeros_like(tx)
        self.v_train['col_point'] = tf.zeros_like(tx)
        
    def add_initial_condition(self, tx_init, u_init, v_init):
        """
        Adding set of initial condition on respective dictionary

        Args:
            tx_init (array_like): initial set of array in temporal and spatial domain
            u_init (array_like): initial condition of u(t,x)/real part
            v_init (array_like): initial condition of v(t,x)/imajinary part
        """        
        self.tx_train['init_point'] = tx_init
        self.u_train['init_point'] = u_init
        self.v_train['init_point'] = v_init
    
    def add_boundary_condition(self, tx_bound, u_bound, v_bound):
        """
        Adding set of boundary condition on respective dictionary

        Args:
            tx_init (array_like): boundary set of array in temporal and spatial domain
            u_init (array_like): boundary condition of u(t,x)/real part
            v_init (array_like): boundary condition of v(t,x)/imajinary part
        """        
        self.tx_train['bound_point'] = tx_bound
        self.u_train['bound_point'] = u_bound
        self.v_train['bound_point'] = v_bound
    
    def initialize(self):
        return self.tx_train, self.u_train, self.v_train
    
    def batch_data(self, batch_size = 100) -> dict:
        """
        Batch x_train and y_train (input data)

        Args:
            tx_train (dict): dictionary that contains the key-value pair of
                collocation point (col_point), initial condition (init_point), 
                boundary condition (bound_point)
            u_train (dict): dictionary that contains the key-value pair of
                real part of collocation point's residue (col_point), initial value (init_point),
                boundary value (bound_point) 
            v_train (dict): dictionary that contains the key-value pair of
                imajinary part of collocation point's residue (col_point), initial value (init_point),
                boundary value (bound_point) 
            batch_size (int or dict[str, int]): 
                If an integer is provided, the same batch size is used for all keys. 
                If a dictionary is provided, each key in `x_train` and `y_train` 
                must have a corresponding batch size specified as a key-value pair. 

        Returns:
            dict: Batched data of x_train and y_train corresponding on each key.
                Each key corresponds to a tuple:`(batched_x, batched_y)`.
        """        
        batched_data = {}
        tx_train = self.tx_train
        u_train = self.u_train
        v_train = self.v_train
        
        common_keys = tx_train.keys() & u_train.keys() & v_train.keys()
        assert common_keys, "No matching keys found between x_train, u_train, and v_train."
        
        for key in common_keys:
            if isinstance(batch_size, dict):
                assert key in batch_size, f"Batch size not specified for the key {key}"
                current_batch_size = batch_size[key]
            elif isinstance(batch_size, int):
                current_batch_size = batch_size
            
            dataset = tf.data.Dataset.from_tensor_slices((tx_train[key], u_train[key], v_train[key]))
            batched_dataset = dataset.batch(current_batch_size).prefetch(tf.data.AUTOTUNE)
            
            batched_data[key] = batched_dataset
        
        return batched_data


