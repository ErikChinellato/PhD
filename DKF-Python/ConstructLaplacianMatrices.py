import numpy as np

def ConstructLaplacianMatrices(N, SamplingTimes):
    """
    Constructs discrete Laplacian matrices for loss function term.
    """
    # Variables
    TimeStep = SamplingTimes[0]  # Assuming constant sampling
    InvTSSq = (1 / TimeStep)**2

    # Assemble the matrices
    L = np.vstack([
        np.concatenate( (np.array([2, -5, 4, -1]), np.zeros(N-4)) ), 
        np.hstack([
            np.atleast_2d(np.concatenate(([1], np.zeros(N-3)))).T, 
            np.diag(-2 * np.ones(N-2)) + np.diag(np.ones(N-3), 1) + np.diag(np.ones(N-3), -1),
            np.atleast_2d(np.concatenate( (np.zeros(N-3), [1]) )).T
        ]),
        np.concatenate( (np.zeros(N-4), np.array([-1, 4, -5, 2])) )
    ])

    # Uncomment if you want to scale L by InvTSSq
    # L = InvTSSq * L
    
    LtL = L.T @ L

    return L, LtL
