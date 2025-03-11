import tensorflow as tf 
class nlseGradient(tf.keras.layers.Layer):
    """
    Custom gradient layer to compute the residue of NLSE derivatives

    Args:
        tf (tf.keras.model): keras model
    """    
    def __init__(self, model, T, L, alpha, beta, gamma):
        super(nlseGradient, self).__init__()     
        self.model = model
        self.T = T 
        self. L = L
        self.gamma = gamma 
        self.beta = beta 
        self.alpha = alpha
    
    @tf.function
    def compute_residue(self, tx: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of input tensor
        """
        with tf.GradientTape() as tape1:
            tape1.watch(tx)
            with tf.GradientTape() as tape2:
                tape2.watch(tx)
                uv = self.model(tx)
                
            duv_dtx = tape2.batch_jacobian(uv, tx, experimental_use_pfor = True)
            duv_dt = duv_dtx[..., 0]  # shape (n, 2) → [du/dt, dv/dt]
        
        d2vu_dt2 = d2uv_dt2 = tape1.batch_jacobian(duv_dt, tx, experimental_use_pfor=True)[..., 0]
        d2vu_dt2 = d2vu_dt2[:, ::-1]
        
        uv_residue = tf.zeros_like(uv, dtype = tf.float32)
        scalar = tf.reduce_sum(tf.square(uv), axis = 1)
        scalar = tf.stack([scalar, scalar], axis=-1)

        uv_residue = duv_dtx[...,1]/self.L + self.alpha*uv/2 - self.beta/(2*self.T**2)*d2vu_dt2 + self.gamma*scalar*uv[:, ::-1]
        
        return uv_residue 

    def compute_labelled_data(self, tx_label: tf.Tensor):
        uv = self.model(tx_label) #compute pulse eq. u(t,x)
        return uv
            