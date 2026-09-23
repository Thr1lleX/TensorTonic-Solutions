def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    xt = x0
    current_step = 0
    while current_step<steps:
        current_step+=1 
        derivative = 2*a*xt+b
        xt = xt-lr*derivative
    return xt