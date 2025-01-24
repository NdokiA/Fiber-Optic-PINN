import tensorflow as tf
from itertools import cycle
import tensorflow_probability as tfp
from tqdm import tqdm
from gradient import nlseGradient

class pinnOptimizer(nlseGradient):
    '''
    Algorithm for processing input and do backpropagation
    '''
    
    def __init__(self, model, batched_dict, alpha, beta2, gamma, penalty = True):
        super().__init__()
        self.model = model
        self.optimizer = tfp.optimizer.lbfgs_minimize
        
        labelled_keys = [key for key in batched_dict.keys() if key != 'col_point']
        self.collocation_data = batched_dict['col_point']
        self.labelled_data = tf.data.Dataset.zip(tuple(batched_dict[key] for key in labelled_keys))
        
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma 

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
        
        u_residue, v_residue, correction = self.compute_residue(collocation_batch[0], self.model,
                                self.gamma, self.beta2, self.alpha, self.penalty)
        
        loss_u = tf.reduce_mean(tf.keras.losses.MSE(u_residue, collocation_batch[1]), axis = 0)
        loss_v = tf.reduce_mean(tf.keras.losses.MSE(v_residue, collocation_batch[2]), axis = 0)
        total_loss = tf.cast(loss_u + loss_v + correction, tf.float32)
        return total_loss

    def _compute_labelled_loss(self, one_labelled_batch) -> tf.Tensor:
        """
        Compute labelled loss of a batch of labelled data

        Args:
            one_labelled_batch (tuple(EagerTensor)): one batch of labelled data contains:
            (tx, u(expected real value), v(expected imaginary value))

        Returns:
            tf.Tensor: scalar tensor of total loss of real and imaginary part of labelled data
        """        
        
        #Tiga proses di bawah perlu diubah ke global untuk seluruh batch labelled data daripada lokal, kelamaan.
        tx_label = tf.concat([batch[0] for batch in one_labelled_batch], axis = 0)
        u_label = tf.concat([batch[1] for batch in one_labelled_batch], axis = 0)
        v_label = tf.concat([batch[2] for batch in one_labelled_batch], axis = 0)
        
        computed_u, computed_v = self.compute_labelled_data(tx_label, self.model)
        
        loss_u = tf.reduce_mean(tf.keras.losses.MSE(computed_u, u_label))
        loss_v = tf.reduce_mean(tf.keras.losses.MSE(computed_v, v_label))
        total_loss = tf.cast(loss_u + loss_v, tf.float32)
        return total_loss


    def _compute_gradient(self, collocation_batch, one_labelled_batch) -> tuple:        
        """

        Args:
            collocation_batch (tuple(EagerTensor)): one batch of collocation point data
            one_labelled_batch (tuple(EagerTensor)): one batch of labelled point data

        Returns:
            current_residue_loss, current_labelled_loss, total_loss, flattened_gradients
        """        
        with tf.GradientTape() as tape:
            current_residue_loss = self._compute_residue_loss(collocation_batch)
            current_labelled_loss = self._compute_labelled_loss(one_labelled_batch)
            total_loss = current_residue_loss+current_labelled_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        flattened_gradients = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return total_loss, flattened_gradients 
    
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
        
    
    def optimize_single_batch(self, collocation_batch, labelled_batch) -> tuple[tf.Tensor]:
        """
        Optimize single batch via tensorflow probability's optimizer LBFGS-minimze.

        Args:
            collocation_batch (tuple(EagerTensor)): one batch of collocation point data
            one_labelled_batch (tuple(EagerTensor)): one batch of labelled point data

        Returns:
            tuple[tf.Tensor]: current_residue_loss, current_labelled_loss
        """    
            
        current_labelled_loss = self._compute_labelled_loss(labelled_batch)
        current_residue_loss = self._compute_residue_loss(collocation_batch)
        
        initial_position = tf.concat([tf.reshape(v,[-1]) for v in self.model.trainable_variables], axis = 0)
        
        def value_and_gradients_function(position):
        # Assign position to the model variables
            self.assign_model_parameters(position)
            
            # Compute the total loss again with updated model parameters
            total_loss, gradients = self._compute_gradient(collocation_batch, labelled_batch)
            
            return total_loss, gradients
        
        
        results = self.optimizer(
            value_and_gradients_function = value_and_gradients_function,
            initial_position = initial_position,
        )
        
        self.assign_model_parameters(results.position)
        return current_residue_loss, current_labelled_loss

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
            labelled_iterator = cycle(self.labelled_data)
            print(f"Epoch {epoch+1}/{epochs}")

            progress_bar = tqdm(range(len(self.collocation_data)), desc="Training Batches", unit="batch")
            
            for _ in progress_bar:
                collocation_batch = next(collocation_iterator)
                labelled_batch = next(labelled_iterator)
                
                batch_residue_loss, batch_labelled_loss = self.optimize_single_batch(collocation_batch, labelled_batch)
                progress_bar.set_description(f"Total Loss: {batch_residue_loss+batch_labelled_loss:.4f}")

                epoch_residue_loss += batch_residue_loss
                epoch_labelled_loss += batch_labelled_loss
                
            
            epoch_residue_loss /= len(self.labelled_data)
            epoch_labelled_loss /= len(self.labelled_data)
            
            print(f'Residue Loss : {epoch_residue_loss:.4f}; Labelled Loss : {epoch_labelled_loss:.4f}')
            self.residue_losses.append(epoch_residue_loss)
            self.labelled_losses.append(epoch_labelled_loss)