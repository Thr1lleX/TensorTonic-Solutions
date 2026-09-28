def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    
    expected_returns = [1]
    W = []
    for k in range(len(returns)):
        if k == 0:
            expected_returns[k] = returns[k]
            W.append(1+returns[k])
        else:
            W.append(W[k-1]* (1+returns[k]))
            expected_returns.append(W[k] - 1)
        
        

    return expected_returns