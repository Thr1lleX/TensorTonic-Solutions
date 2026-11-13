import numpy as np

def compute_ddpm_loss(epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean DDPM noise-prediction loss.
    """
    N = len(epsilon)

    loss = np.mean((np.asarray(epsilon) - np.asarray(epsilon_pred))**2, axis = 1)
    return float(np.mean(loss))