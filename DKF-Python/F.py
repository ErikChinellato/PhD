from scipy.optimize import fsolve
import numpy as np

from DeepKalmanFilter.DynamicEquation import *

def F(x, p, u, s, Layer, NetParameters):
    """
    Computes the state prediction xp.
    """
    # Variables
    Model = NetParameters['Model']

    # Equation for the dynamics
    def DynEq(z):
        return DynamicEquation(z, x, p, u, s, Model['M'], Model['K'], Model['D'][:,Layer:Layer+1], Model['SamplingTimes'][Layer], NetParameters)

    # Solve for the state prediction
    options = {'xtol': 1e-6, 'maxfev': 1000000}
    xp = np.atleast_2d( fsolve(DynEq, x, xtol=options['xtol'], maxfev=options['maxfev']) ).T
    return xp
