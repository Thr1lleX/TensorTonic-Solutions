import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    element = np.asarray(x, dtype=float) 
    element = -element
    
    return np.array(1/(1+np.exp(element)))