import numpy as np

def linear_beta_schedule(T: int,
                         beta_1: float = 0.0001,
                         beta_T: float = 0.02) -> list[float]:
    """
    Returns T linearly spaced beta values.
    """
    beta = [beta_1]
    for t in range(2,T):
        beta.append(beta_1 + (t - 1)/(T - 1) * (beta_T - beta_1))

    if T == 1:
        return np.round(beta,6).tolist()
    else:
        beta.append(beta_T)
    
        return np.round(beta,6).tolist()