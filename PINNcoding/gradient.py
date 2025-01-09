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

        Args:
            tx (tf.Tensor): Compute 1st and 2nd derivatives of h(t,x)
            using Jacobian Matrix
        Outpus:
            (u), (v)
        """        
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(tx) 
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(tx)
                uv = self.model(tx) #compute pulse eq. u(t,x)
                u = tf.reshape(uv[..., 0], (-1, 1))
                v = tf.reshape(uv[..., 1], (-1, 1)) 
            
            du_dtx = tape2.batch_jacobian(u, tx)
            du_dt = du_dtx[..., 0]
            du_dx = du_dtx[..., 1]
            
            dv_dtx = tape2.batch_jacobian(v, tx)
            dv_dt = dv_dtx[..., 0]
            dv_dx = dv_dtx[..., 1]
        
        d2u_dt2 = tape1.batch_jacobian(du_dt, tx)[...,0]
        d2u_dx2 = tape1.batch_jacobian(du_dx, tx)[...,1]
        
        d2v_dt2 = tape1.batch_jacobian(dv_dt, tx)[...,0]
        d2v_dx2 = tape1.batch_jacobian(dv_dx, tx)[...,1]
        
        del tape1, tape2
        return ((u, du_dt, du_dx, d2u_dt2, d2u_dx2), (v, dv_dt, dv_dx, d2v_dt2, d2v_dx2))