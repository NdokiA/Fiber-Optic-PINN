import tensorflow as tf

import tensorflow_probability as tfp
from tqdm import tqdm
from pinnPack.gradient import nlseGradient

import time as Time
from functools import wraps


class pinnOptimizer(nlseGradient):
    '''
    Algorithm for processing input and do backpropagation
    '''
    
    def __init__(self, model, collocation_data, labelled_data ,
                 T, L, alpha, beta2, gamma):
        super().__init__(model, T, L, alpha, beta2, gamma)
        self.model = model
        
        self.ADAM = tf.optimizers.Adam(learning_rate = 1e-3, jit_compile=False)
        self.LBFGS = tfp.optimizer.lbfgs_minimize
        
        self.labelled_data = labelled_data
        self.collocation_data = collocation_data
        
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma 
        
        self.T = T 
        self.L = L
        
        self.loss_records = []
        self.time_records = []
    
    def _loss_fn(self, collocation_data, labelled_data) -> tf.Tensor:
        
        uv_residue = self.compute_residue(collocation_data[0])
        uv_data = self.compute_labelled_data(labelled_data[0])
        
        residual_loss = tf.reduce_mean(tf.keras.losses.MSE(uv_residue, collocation_data[1]))
        data_loss = tf.reduce_mean(tf.keras.losses.MSE(uv_data, labelled_data[1]))
                                             
        losses = tf.cast(residual_loss + data_loss, tf.float32)
        return losses

    def _gradient(self, collocation_data, labelled_data) -> tuple:        
        with tf.GradientTape() as tape:
            loss = self._loss_fn(collocation_data, labelled_data)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        flattened_gradients = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return loss, flattened_gradients 
    
    @tf.function
    def assign_model_parameters(self, position):
        
        #Compute sizes of trainable variables
        sizes = tf.convert_to_tensor([tf.size(v) for v in self.model.trainable_variables])
        
        # Unflatten the parameters and assign back to the model variables
        model_vars = tf.split(position, sizes)
        
        for var, opt_var in zip(self.model.trainable_variables, model_vars):
            var.assign(tf.reshape(opt_var, var.shape))

    def optimize_adam(self, collocation_data, labelled_data) -> float: 
        
        with tf.GradientTape() as tape:
            loss = self._loss_fn(collocation_data, labelled_data)
    
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.ADAM.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss
    
    def optimize_lbfgs(self, collocation_data, labelled_data) -> float:
            
        loss = self._loss_fn(collocation_data, labelled_data)
        
        initial_position = tf.concat([tf.reshape(v,[-1]) for v in self.model.trainable_variables], axis = 0)
        
        def value_and_gradients_function(position):
        # Assign position to the model variables
            self.assign_model_parameters(position)
            
            # Compute the total loss again with updated model parameters
            loss, gradients = self._gradient(collocation_data, labelled_data)
            
            return loss, gradients
        
        
        results = self.LBFGS(
            value_and_gradients_function = value_and_gradients_function,
            initial_position = initial_position,
            tolerance = 1e-5,
            max_iterations = 20,
        )
        
        self.assign_model_parameters(results.position)
        return loss

    def fit(self, adam_epochs, bfgs_epochs):
        """
        Optimize the model for each epoch

        Args:
            epochs (int): number of epoch/iteration.
        """        
        adam_bar = tqdm(range(adam_epochs), desc='Adam_Epochs', unit='epoch')
        bfgs_bar = tqdm(range(bfgs_epochs), desc='LBFGS_Epochs', unit='epoch')
        total_time = 0
        
        print('Processing using ADAM Optimalization strategies')
        for epoch in adam_bar:
            start = Time.time()
            loss = self.optimize_adam(self.collocation_data, self.labelled_data)
            stop = Time.time()
            
            adam_bar.set_postfix(loss=f"{loss:.4e}")
            total_time += (stop-start)
            self.time_records.append(total_time)            
            self.loss_records.append(loss)
        
        print('Processing using L-BFGS Optimalization strategies')
        for epoch in bfgs_bar:
            start = Time.time()
            loss = self.optimize_lbfgs(self.collocation_data, self.labelled_data)
            stop = Time.time()
            
            bfgs_bar.set_postfix(loss=f"{loss:.4e}")
            total_time += (stop-start)
            self.time_records.append(total_time)  
            self.loss_records.append(loss)