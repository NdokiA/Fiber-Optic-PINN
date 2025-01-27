import tensorflow as tf
from itertools import cycle
import tensorflow_probability as tfp
from tqdm import tqdm
from gradient import nlseGradient

class pinnOptimizer(nlseGradient):
    '''
    Algorithm for processing input and do backpropagation
    '''
    
    def __init__(self, model, batched_dict ,T, L, alpha, beta2, gamma, penalty):
        super().__init__()
        self.model = model
        self.optimizer = tfp.optimizer.lbfgs_minimize
        
        self.labelled_data = batched_dict['labelled_data']
        self.collocation_data = batched_dict['col_point']
        
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma 
        
        self.T = T 
        self.L = L

        self.penalty = penalty
        
        self.residue_losses = []
        self.labelled_losses = []
    
    def _compute_residue_loss(self, collocation_batch) -> tf.Tensor:
        """_
        Compute the residue loss of a batch of collocation point

        Args:
            collocation_batch (tuple(EagerTensor)): one batch of collocation point contains:
            (col_point(tx), u(expected real residue), v(expected imaginary residue))

        Returns:
            tf.Tensor: scalar tensor of the total residue loss of real and imaginary part
        """        
        
        u_residue, v_residue, correction = self.compute_residue(
                                                    collocation_batch[0], self.model,
                                                    self.T, self.L,
                                                    self.gamma, self.beta2, self.alpha, self.penalty
                                                    )
        
        loss_u = tf.reduce_mean(tf.keras.losses.MSE(u_residue, collocation_batch[1]), axis = 0)
        loss_v = tf.reduce_mean(tf.keras.losses.MSE(v_residue, collocation_batch[2]), axis = 0)
        total_loss = tf.cast(loss_u + loss_v + correction, tf.float32)
        return total_loss

    def _compute_labelled_loss(self, labelled_batch) -> tf.Tensor:
        """
        Compute labelled loss of a batch of labelled data

        Args:
            one_labelled_batch (tuple(EagerTensor)): one batch of labelled data contains:
            (tx, u(expected real value), v(expected imaginary value))

        Returns:
            tf.Tensor: scalar tensor of total loss of real and imaginary part of labelled data
        """        
    
        computed_u, computed_v = self.compute_labelled_data(labelled_batch[0], self.model)
        
        loss_u = tf.reduce_mean(tf.keras.losses.MSE(computed_u, labelled_batch[1]))
        loss_v = tf.reduce_mean(tf.keras.losses.MSE(computed_v, labelled_batch[2]))
        total_loss = tf.cast(loss_u + loss_v, tf.float32)
        return total_loss


    def _compute_gradient(self, batch, is_residue) -> tuple:        
        """

        Args:
            batch: one batch of data
            is_residue: check whether to use residue or labelled loss

        Returns:
            loss, flattened gradients
        """        
        func = self._compute_residue_loss if is_residue else self._compute_labelled_loss
        with tf.GradientTape() as tape:
            loss = func(batch)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        flattened_gradients = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return loss, flattened_gradients 
    
    def assign_model_parameters(self, position):
        """
        Assign the flattened position vector back to the model's trainable variables.

        Args:
            position (tf.Tensor): flattened tensor of model parameters.
        """
        # Unflatten the parameters and assign back to the model variables
        model_vars = tf.split(position, [tf.size(v) for v in self.model.trainable_variables])
        
        for var, opt_var in zip(self.model.trainable_variables, model_vars):
            var.assign(tf.reshape(opt_var, var.shape))
        
    
    def optimize_single_batch(self, batch, is_residue) -> tuple[tf.Tensor]:
        """
        Optimize single batch via tensorflow probability's optimizer LBFGS-minimze.

        Args:
            batch: one batch of data
            is_residue: check whether to use residue or labelled loss

        Returns:
            current_loss
        """    
            
        loss = self._compute_residue_loss(batch) if is_residue else self._compute_labelled_loss(batch)
        
        initial_position = tf.concat([tf.reshape(v,[-1]) for v in self.model.trainable_variables], axis = 0)
        
        def value_and_gradients_function(position):
        # Assign position to the model variables
            self.assign_model_parameters(position)
            
            # Compute the total loss again with updated model parameters
            total_loss, gradients = self._compute_gradient(batch, is_residue)
            
            return total_loss, gradients
        
        
        results = self.optimizer(
            value_and_gradients_function = value_and_gradients_function,
            initial_position = initial_position,
        )
        
        self.assign_model_parameters(results.position)
        return loss

    def fit(self, epochs):
        """
        Optimize the model for each epoch

        Args:
            epochs (int): number of epoch/iteration.
        """        
        for epoch in range(epochs):
            epoch_residue_loss = 0
            epoch_labelled_loss = 0
            
            collocation_iterator = iter(self.collocation_data)
            labelled_iterator = iter(self.labelled_data)
            print(f"Epoch {epoch+1}/{epochs}")

            collocation_bar = tqdm(range(len(self.collocation_data)), desc="Training Batches", unit="batch")
            labelled_bar = tqdm(range(len(self.labelled_data)), desc="Labelled Batches", unit="batch")
            
            for _ in collocation_bar:
                collocation_batch = next(collocation_iterator)
                residue_loss = self.optimize_single_batch(collocation_batch, is_residue = True)
                collocation_bar.set_description(f"Residue Loss: {residue_loss:.4e}")
                epoch_residue_loss += residue_loss
                
            for _ in labelled_bar:
                labelled_batch = next(labelled_iterator)
                labelled_loss = self.optimize_single_batch(labelled_batch, is_residue = False)
                labelled_bar.set_description(f"Labelled Loss: {labelled_loss:.4e}")
                epoch_labelled_loss += labelled_loss
            
            
            epoch_residue_loss /= len(self.collocation_data)
            epoch_labelled_loss /= len(self.labelled_data)
            
            print(f'Residue Loss : {epoch_residue_loss:.4f}; Labelled Loss : {epoch_labelled_loss:.4f}')
            self.residue_losses.append(epoch_residue_loss)
            self.labelled_losses.append(epoch_labelled_loss)