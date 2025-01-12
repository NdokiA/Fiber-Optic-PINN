import tensorflow as tf 
class nlseGradient(tf.keras.layers.Layer):
    """
    Custom gradient layer to compute the residue of NLSE derivatives

    Args:
        tf (tf.keras.model): keras model
    """    
    def __init__(self, model: tf.keras.Model):
        super(nlseGradient, self).__init__()    
        self.model = model    
    
    @tf.function
    def call(self, tx: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of input tensor
        """
        with tf.GradientTape() as tape1:
            tape1.watch(tx)
            with tf.GradientTape() as tape2:
                tape2.watch(tx)
                uv = self.model(tx)
                
            duv_dtx = tape2.batch_jacobian(uv, tx, experimental_use_pfor=True)
            
        d2uv_dtx2 = tape1.batch_jacobian(duv_dtx, tx, experimental_use_pfor=True)
        d2uv_dtx2 = tf.reshape(d2uv_dtx2, (tf.shape(d2uv_dtx2)[0],2,4))

        # Split the results into u and v
        u, v = [tf.cast(t, tf.float32) for t in tf.split(uv, num_or_size_splits=2, axis=-1)]
        du_dt, du_dx = [tf.cast(t, tf.float32) for t in tf.split(duv_dtx[..., 0], num_or_size_splits=2, axis=-1)]
        dv_dt, dv_dx = [tf.cast(t, tf.float32) for t in tf.split(duv_dtx[..., 1], num_or_size_splits=2, axis=-1)]
        d2u_dt2,_, _, d2u_dx2 = [tf.cast(t, tf.float32) for t in tf.split(d2uv_dtx2[:, 0], num_or_size_splits=4, axis=-1)]
        d2v_dt2, _, _, d2v_dx2 = [tf.cast(t, tf.float32) for t in tf.split(d2uv_dtx2[:, 1], num_or_size_splits=4, axis=-1)]


        return ((u, du_dt, du_dx, d2u_dt2, d2u_dx2), (v, dv_dt, dv_dx, d2v_dt2, d2v_dx2))


    
    def compute_residue(self, tx: tf.Tensor, gamma: float, beta: float, alpha: float):
        (u, du_dt, du_dx, d2u_dt2, d2u_dx2), (v, dv_dt, dv_dx, d2v_dt2, d2v_dx2) = self.call(tx)
        scalar = u**2+v**2
        u_residue = du_dx + alpha/2*u - beta/2*d2v_dt2 + gamma*scalar*v
        v_residue = dv_dx + alpha/2*v + beta/2*d2u_dt2 - gamma*scalar*u
        
        return u_residue, v_residue
    
    def compute_labelled_data(self, tx_label: tf.Tensor):
        uv = self.model(tx_label) #compute pulse eq. u(t,x)
        u_label = tf.reshape(uv[..., 0], (-1, 1))
        v_label = tf.reshape(uv[..., 1], (-1, 1)) 
        
        return u_label, v_label
            