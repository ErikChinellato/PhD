import numpy as np

def ConstructTensorizedGains(NetWeights, NetParameters):
    """
    Inserts the Kalman gains into a 3-D tensor.
    """
    SharedWeights = NetParameters['SharedWeights']
    Layers = NetParameters['Layers']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']

    TensorizedGains = np.zeros((StateDimension, ObservationDimension, Layers))

    if SharedWeights == 'No':
        # Assemble tensor
        for Layer in range(Layers):
            TensorizedGains[:,:,Layer] = NetWeights[Layer]
    
    if SharedWeights == 'Yes':
        # Do nothing
        pass
    
    return TensorizedGains
