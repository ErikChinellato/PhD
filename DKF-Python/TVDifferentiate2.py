import numpy as np

def TVDifferentiate2(h, alpha, B, D, AtA, Atf, uEst, maxit):
    """
    Computes u = f' using TV normalization. f must be a column vector.
    """
    epsilon = 1e-8

    for _ in range(maxit):
        # Compute the L matrix
        DuEst = D @ uEst
        L = alpha * h * D.T @ np.diag(1. / (np.sqrt((0.5 * DuEst) ** 2 + epsilon))) @ D
        
        # Compute the Hessian and gradient
        H = L + AtA
        g = -(AtA @ uEst - Atf + L @ uEst)
        
        # Update uEst
        uEst += np.linalg.solve(H, g)
    
    # Return denoised f
    fEst = B @ uEst
    return uEst, fEst