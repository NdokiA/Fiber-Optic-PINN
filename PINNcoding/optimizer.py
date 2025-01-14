import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from gradient import nlseGradient

class pinnOptimizer(nlseGradient):
    '''
    Algorithm for processing input and do backpropagation
    '''
    
    def __init__(self, model, batched_dict, alpha, beta2, gamma):
        super().__init__(model)
        self.model = model
        self.optimizer = tfp.optimizer.lbfgs_minimize
        
        labelled_keys = [key for key in batched_dict.keys() if key != 'col_point']
        self.collocation_data = batched_dict['col_point']
        self.labelled_data = tf.data.Dataset.zip(tuple(batched_dict[key] for key in labelled_keys))
        
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma 
        
        self.residue_losses = []
        self.labelled_losses = []
    
    def _compute_residue_loss(self, collocation_batch):
        
        u_residue, v_residue = self.compute_residue(collocation_batch[0], 
                                self.gamma, self.beta2, self.alpha)
        
        loss_u = tf.reduce_mean(tf.keras.losses.MSE(u_residue, collocation_batch[1]), axis = 0)
        loss_v = tf.reduce_mean(tf.keras.losses.MSE(v_residue, collocation_batch[2]), axis = 0)
        total_loss = tf.cast(loss_u + loss_v, tf.float32)
        return total_loss

    def _compute_labelled_loss(self, one_labelled_batch):
        
        tx_label = tf.concat([batch[0] for batch in one_labelled_batch], axis = 0)
        u_label = tf.concat([batch[1] for batch in one_labelled_batch], axis = 0)
        v_label = tf.concat([batch[2] for batch in one_labelled_batch], axis = 0)
        
        computed_u, computed_v = self.compute_labelled_data(tx_label)
        
        loss_u = tf.reduce_mean(tf.keras.losses.MSE(computed_u, u_label))
        loss_v = tf.reduce_mean(tf.keras.losses.MSE(computed_v, v_label))
        total_loss = tf.cast(loss_u + loss_v, tf.float32)
        return total_loss

    def _compute_gradient(self, current_residue_loss, current_labelled_loss):        
        with tf.GradientTape() as tape:
            total_loss = current_residue_loss+current_labelled_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        flattened_gradients = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return total_loss, flattened_gradients 
    
    def optimize_single_batch(self, collocation_batch, labelled_batch):
        current_residue_loss = self._compute_residue_loss(collocation_batch)
        current_labelled_loss = self._compute_labelled_loss(labelled_batch)
        results = self.optimizer(
            value_and_gradients_function = self._compute_gradient(current_residue_loss, current_labelled_loss),
            initial_position = tf.concat([tf.reshape(v,[-1]) for v in self.model.trainable_variables], axis = 0)
        )
        
        optimized_vars = tf.split(results.position, [tf.size(v) for v in self.model.trainable_variables])
        
        for var, opt_var in zip(self.model.trainable_variables, optimized_vars):
            var.assign(tf.reshape(opt_var, var.shape))
        
        return current_residue_loss, current_labelled_loss

    def fit(self, epochs):
        for epoch in epochs:
            epoch_residue_loss = 0
            epoch_labelled_loss = 0
            
            print(f"Epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(range(len(self.labelled_data)), desc="Training Batches", unit="batch")
            for _ in progress_bar:
                collocation_batch = next(iter(self.collocation_data))
                labelled_batch = next(iter(self.labelled_data))
                
                batch_residue_loss, batch_labelled_loss = self.optimize_single_batch(collocation_batch, labelled_batch)
                progress_bar.set_description(f"Total Loss: {batch_residue_loss+batch_labelled_loss:.4f}")

                epoch_residue_loss += batch_residue_loss
                epoch_labelled_loss += batch_labelled_loss
                
            
            epoch_residue_loss /= len(self.labelled_data)
            epoch_labelled_loss /= len(self.labelled_data)
            
            print(f'Residue Loss : {epoch_residue_loss:.4f}; Labelled Loss : {epoch_labelled_loss:.4f}')
            self.residue_losses.append(epoch_residue_loss)
            self.labelled_losses.append(epoch_labelled_loss)