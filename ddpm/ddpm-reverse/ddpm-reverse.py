import numpy as np

def reverse_step(x_t: list, t: int, epsilon_pred: list, betas: list[float], z: list = None) -> list:
    """
    Returns x at timestep t - 1, rounded to four decimals.
    """

    beta_t = betas[t-1]
    alpha_t = 1 - beta_t
    alpha_t_bar = np.cumprod(np.ones(len(betas))-betas)
    

    mean = 1/(np.sqrt(alpha_t)) * (x_t - beta_t/np.sqrt(1 - alpha_t_bar[t-1]) * np.asarray(epsilon_pred))

    if t == 1:
        return np.round(mean, 4)

    else:
        if z is not None:
            return np.round(mean + np.sqrt(betas[t-1])*np.asarray(z),4)
        else:
            raise Exception("No z provided")