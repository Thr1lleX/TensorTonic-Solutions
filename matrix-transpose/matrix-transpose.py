import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    mat = np.asarray(A)
    m,n = mat.shape
    transpose = np.zeros((n,m))
    for i in range(0,m):
        for j in range (0,n):
            transpose[j,i] = mat[i,j]
    return transpose
