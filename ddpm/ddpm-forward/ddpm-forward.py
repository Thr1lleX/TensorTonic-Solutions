import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    ones = np.ones(len(betas))
    betas = np.asarray(betas)
    return np.cumprod(ones-betas, axis = 0)

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """
    
    x_0 = np.asarray(x_0)
    epsilon = np.asarray(epsilon)
    
    sqrt_alpha_bar = np.sqrt(get_alpha_bar(betas)[t-1]) * x_0
    one_minus_alpha_bars = np.sqrt(1 - get_alpha_bar(betas)[t-1]) * epsilon

    x_t = sqrt_alpha_bar + one_minus_alpha_bars
    return x_t