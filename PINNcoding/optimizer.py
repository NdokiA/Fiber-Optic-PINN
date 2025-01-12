import tensorflow as tf
import tensorflow_probability as tfp
from gradient import nlseGradient

class pinnOptimizer(nlseGradient):
    '''
    Algorithm for processing input and do backpropagation
    '''
    
    def __init__(self, model, batched_dict, alpha, beta2, gamma):
        super().__init__()
        self.model = model
        self.optimizer = tfp.optimizer.lbfgs_minimize
        
        labelled_keys = [key for key in batched_dict.keys() if key != 'col_point']
        self.collocation_data = batched_dict['col_point']
        self.labelled_data = tf.data.Dataset.zip(tuple(batched_dict[key] for key in labelled_keys))
        
        self.alpha = alpha
        self.beta2 = beta2
        self.gamma = gamma 
        
        self.current_residue_loss = 0
        self.current_labelled_loss = 0
    
    def _compute_residue_loss(self, collocation_batch):
        
        u_residue, v_residue = self.compute_residue(collocation_batch[0], 
                                self.gamma, self.beta2, self.alpha)
        
        loss_u = tf.keras.losses.MSE(u_residue, collocation_batch[1])
        loss_v = tf.keras.losses.MSE(v_residue, collocation_batch[2])
        return loss_u+loss_v

    def _compute_labelled_loss(self, all_labelled_batch):
        loss_u, loss_v = 0, 0
        for labelled_batch in all_labelled_batch:
            u_label, v_label = self.compute_labelled_data(labelled_batch[0])
        
            loss_u += tf.keras.losses.MSE(u_label, labelled_batch[1])
            loss_v += tf.keras.losses.MSE(v_label, labelled_batch[2])
        
        return loss_u+loss_v

    def _compute_loss_gradient(self, collocation_batch, labelled_batch):        
        with tf.GradientTape() as tape:
            self.current_residue_loss = self._compute_residue_loss(collocation_batch)
            self.current_labelled_loss = self._compute_labelled_loss(labelled_batch)
            total_loss = self.current_residue_loss+self.current_labelled_loss
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        flattened_gradients = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return total_loss, flattened_gradients 
    
    def optimize_single_batch(self, collocation_batch, labelled_batch):
        results = self.optimizer(
            value_and_gradients_function = self._compute_loss_gradient(collocation_batch, labelled_batch),
            initial_position = tf.concat([tf.reshape(v,[-1]) for v in self.model.trainable_variables], axis = 0)
        )
        
        optimized_vars = tf.split(results.position, [tf.size(v) for v in self.model.trainable_variables])
        
        for var, opt_var in zip(self.model.trainable_variables, optimized_vars):
            var.assign(tf.reshape(opt_var, var.shape))
