import tensorflow as tf 
import numpy as np

class initData:
    '''
    initData class is used to initialize the collocation points and labelled data using random LHS sampling
    '''
    def __init__(self):
        pass 

    def lhs_sampling(self, n: int, d: int, seed: int = None) -> np.ndarray:
        """
        Latin Hypercube Sampling (LHS) to generate random points

        Args:
            n (int): Number of samples
            d (int): Dimension of samples
            seed (int, optional): Random seed. Defaults to None

        Returns:
            np.ndarray: Random samples
        """        
        rng = np.random.default_rng(seed)
        result = np.zeros((n,d))

        for i in range(d):
            result[:,i] = rng.permutation(np.linspace(0,1,n,endpoint=False)) + rng.random((n))/n
        
        return result
    
    def init_points(self, final: np.ndarray, start: np.ndarray, n: int, seed: int = None) -> np.ndarray:
        '''
        Initialize points using Latin Hypercube Sampling
        
        Args:
            final (np.ndarray): Expected final point (have to have the same dimension with initial)
            initial (np.ndarray): Expected initial point (have to have the same dimension with final)
            n (int): Number of samples
            seed (int, optional): Random seed. Defaults to None.
        '''
        d = final.shape[-1] if isinstance(final, np.ndarray) else 1
        col_points = self.lhs_sampling(n,d,seed)*(final-start) + start
        return col_points
    
class batchData:
    '''
    batchData class is used to store collocation and labelled data and batch for training purposes
    '''
    def __init__(self):
        self.tx_train = {
            'col_point': None,
        }
        self.uv_train = {
            'col_point': None,
        }
    
    def add_collocation_point(self,tx):
        """
        Adding set of collocation points to self.tx_train, and
        expected residual value on self.u_train and self.v_train via tf.zeros_like

        Args:
            tx (array_like): Inirialized collocation point
        """        
        self.tx_train['col_point'] = tx
        self.uv_train['col_point'] = tf.zeros((tx.shape[0],2))
    
    def add_labelled_data(self, tx, u, v, label):
        """
        Adding set of labelled data to self.tx_train, self.u_train, and self.v_train

        Args:
            tx (array_like): Initialized labelled data
            u (array_like): real part of labelled data
            v (array_like): imaginary part of labelled data
        """        
        self.tx_train[label] = tx
        self.uv_train[label] = tf.concat([u,v], axis = -1)
        
   
    def get_keys(self)->list:
        """
        Get the keys of dataset's dictionary

        Returns:
            list: List of keys
        """        
        return list(self.tx_train.keys())
        
    def batch_data(self, batch_size = 100) -> dict:
        """
        Batch x_train and y_train (input data)

        Args:
            batch_size (int or dict[str, int]): Batch size for each key 

        Returns:
            dictionary of batched data corresponding to each key
        """        
        batched_data = {}
        common_keys = self.get_keys()
        
        labelled_keys = list(set(common_keys) - set(['col_point']))
        tx_labelled = np.concatenate([self.tx_train[key] for key in labelled_keys], axis = 0)
        uv_labelled = np.concatenate([self.uv_train[key] for key in labelled_keys], axis = 0)
        
        collocation_data = tf.data.Dataset.from_tensor_slices((
            tf.cast(self.tx_train['col_point'], tf.float32), 
            tf.cast(self.uv_train['col_point'], tf.float32)
        ))

        labelled_data = tf.data.Dataset.from_tensor_slices((
            tf.cast(tx_labelled, tf.float32), 
            tf.cast(uv_labelled, tf.float32)
        ))
        
        collocation_data = collocation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        labelled_data = labelled_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        batched_data['col_point'] = collocation_data
        batched_data['labelled_data'] = labelled_data
        
        return batched_data


