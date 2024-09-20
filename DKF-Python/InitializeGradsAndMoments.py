import numpy as np
import copy
def InitializeGradsAndMoments(NetWeights, NetParameters):
    """
    Initializes the gradients for the net's parameters to zero.
    """
    Layers = NetParameters['Layers']
    SharedWeights = NetParameters['SharedWeights']
    HiddenDynamicsNumber = NetParameters['HiddenDynamicsNumber']

    # Setup gradients
    if SharedWeights == 'No':
        Grads = [None] * (Layers + 1)

        for Layer in range(Layers):
            Grads[Layer] = np.zeros_like(NetWeights[Layer]).astype('float64')
        
        Grads[Layers] = [None] * (HiddenDynamicsNumber + 1)
        for Dyn in range(HiddenDynamicsNumber):
            Grads[Layers][Dyn] = np.zeros_like(NetWeights[Layers][Dyn]).astype('float64')
        Moment1 = copy.deepcopy(Grads)
        Moment2 = copy.deepcopy(Grads)

    if SharedWeights == 'Yes':
        Grads = [None] * (2)

        Grads[0] = np.zeros_like(NetWeights[0]).astype('float64')

        Grads[1] = [None] * (HiddenDynamicsNumber + 1)
        for Dyn in range(HiddenDynamicsNumber):
            Grads[1][Dyn] = np.zeros_like(NetWeights[1][Dyn]).astype('float64')
        Moment1 = copy.deepcopy(Grads)
        Moment2 = copy.deepcopy(Grads)

    return Grads, Moment1, Moment2
