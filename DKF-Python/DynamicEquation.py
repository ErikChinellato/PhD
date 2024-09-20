import numpy as np

from DeepKalmanFilter.ConstructDictionary import *

def DynamicEquation(z, x, p, u, s, M, K, d, Ts, NetParameters):
    """
    Encodes the (implicit) equation for the dynamics to be solved forward in time.
    """
    # Variables
    Experiment = NetParameters['Experiment']

    z = np.atleast_2d(z).T

    # Dictionary for model discovery
    Phi = ConstructDictionary(z, NetParameters)

    if Experiment == '1':
        F = M @ (z - x) - Ts * (K @ z + d + s.T @ Phi.T)

    elif Experiment == '2':
        F = M @ (z - x) - Ts * (K @ z + d + s.T @ Phi.T)

    elif Experiment == '3':
        F = M @ (z - x) - Ts * (K @ z + np.array([0, -z[0]*z[2], z[0]*z[1]]) + d + s.T @ Phi.T)

    elif Experiment == '4':
        F = ( M + np.diag([np.squeeze(p)] + [0] * (2)) ) @ (z - x) - Ts * (K @ z + np.array( [[0], [-z[0,0]*z[2,0]], [z[0,0]*z[1,0]]] ) + d + s.T @ Phi.T)

    elif Experiment == '5':
        K[1, 0] = K[1, 0] + p
        F = M @ (z - x) - Ts * (K @ z + np.array([0, -z[0]*z[2], z[0]*z[1]]) + d + s.T @ Phi.T)

    elif Experiment == '6':
        F = M @ (z - x) - Ts * (K @ z + d + s.T @ Phi.T)

    return np.squeeze(F)
