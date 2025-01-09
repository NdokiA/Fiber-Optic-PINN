import tensorflow as tf 

class physicsNetwork:
  '''
  Build PINN model for the heat equation
  '''
  @classmethod
  def build(cls, num_inputs = 2, layers = [32,32,32,32,32],
            activation = 'tanh', num_outputs = 2):
    '''
    Build PINN model for the heat equation with input shape (t,x)
    output shape u(t,x)
    Inputs:
      num_inputs: number of input variables
      layers: hidden layers
      activation: hidden layers activation function
      num_outputs: number of output variables
    Outputs:
      keras model
    '''
    inputs = tf.keras.layers.Input(shape = (num_inputs,))
    #hidden layers
    x = inputs
    for layer in layers:
      x = tf.keras.layers.Dense(layer, activation = activation)(x)
    #output layer
    outputs = tf.keras.layers.Dense(num_outputs)(x)
    model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
    return model