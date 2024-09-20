import numpy as np

def InitializeDynamics(HiddenDynamicsDimension, Model, Experiment):
    """
    Initializes the dynamics parameters.
    """
    if Experiment == '4':
        Dynamic = np.zeros((HiddenDynamicsDimension, 1))
    return Dynamic
