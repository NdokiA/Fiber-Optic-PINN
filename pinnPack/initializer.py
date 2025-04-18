import torch
import numpy as np

class genData:
    '''
    initData class is used to initialize the collocation points and labelled data using defined sampling technique
    '''
    def __init__(self):
        pass 
    
    @staticmethod
    def lhs_sampling(n: int, d: int, seed: int = 42) -> np.ndarray:
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
    
    @staticmethod
    def sobol_sampling(n: int, d: int, seed: int = None) -> np.ndarray:
        """
        Sobol Sampling using torch.quasirandom

        Args:
            n (int): Number of samples
            d (int): Dimensions
            seed (int, optional): Seed for reproducibility

        Returns:
            np.ndarray: Sobol samples scaled to [0, 1]^d
        """
        if seed is not None:
            torch.manual_seed(seed)

        sobol = torch.quasirandom.SobolEngine(dimension=d)
        samples = sobol.draw(n)  
        return samples.numpy()
    
    @staticmethod
    def generate_points(final: np.ndarray, start: np.ndarray, n: int, seed: int = None) -> np.ndarray:
        '''
        Initialize points using Latin Hypercube Sampling
        
        Args:
            final (np.ndarray): Expected final point (have to have the same dimension with initial)
            initial (np.ndarray): Expected initial point (have to have the same dimension with final)
            n (int): Number of samples
            seed (int, optional): Random seed. Defaults to None.
        '''
        d = final.shape[-1] if isinstance(final, np.ndarray) else 1
        col_points = genData.sobol_sampling(n,d,seed)*(final-start) + start
        return col_points
    
    @staticmethod
    def batch_data(tx_data, uv_data):
        """
        Batch x_train and y_train (input data) into a dataset
        """        
        
        dataset = (
            torch.tensor(tx_data, dtype=torch.float32),
            torch.tensor(uv_data, dtype=torch.float32)
        )

        return dataset


