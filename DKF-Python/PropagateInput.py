import numpy as np

def PropagateInput(Inputs, Measurements, FirstState, Dynamic, F, NetWeights, NetParameters):
    """
    Propagates the Inputs vector (u) and Measurements vector (y) through the network. 
    They are lists of size (1, NetParameters['Layers']).
    F is the 'VARMION' function block. The output is the States vector (x),
    a list of size (1, NetParameters['Layers'] + 1). States[0] is given as an input. 
    Additional outputs MeasurementMinusCFs, GainMeasurementMinusCFs, MeasurementMinusCStates, 
    and FStateDynInputs are saved for later efficiency during backpropagation.
    """

    Layers = NetParameters['Layers']
    C = NetParameters['C']
    SharedWeights = NetParameters['SharedWeights']

    # Setup output
    States = [None] * (Layers + 1)
    MeasurementMinusCStates = [None] * Layers
    GainMeasurementMinusCFs = [None] * Layers

    MeasurementMinusCFs = [None] * Layers
    FStateDynInputs = [None] * Layers

    # Initialize the first state
    States[0] = FirstState

    # Propagate through layers
    for Layer in range(Layers):
        if SharedWeights == 'Yes':
            Indx = 0 
        else:
            Indx = Layer

        # Compute FStateDynInput using the provided function F
        FStateDynInput = F(States[Layer], NetWeights[-1][Dynamic-1], Inputs[:,Layer:Layer+1], NetWeights[-1][-1], Layer, NetParameters)

        # Calculate MeasurementMinusCF and GainMeasurementMinusCF
        MeasurementMinusCF = Measurements[:,Layer+1:Layer+2] - C @ FStateDynInput
        GainMeasurementMinusCF = NetWeights[Indx]@MeasurementMinusCF

        # Save outputs
        States[Layer + 1] = FStateDynInput + GainMeasurementMinusCF
        MeasurementMinusCStates[Layer] = Measurements[:,Layer+1:Layer+2] - C@States[Layer+1]
        GainMeasurementMinusCFs[Layer] = GainMeasurementMinusCF

        MeasurementMinusCFs[Layer] = MeasurementMinusCF
        FStateDynInputs[Layer] = FStateDynInput

    return States, MeasurementMinusCStates, GainMeasurementMinusCFs, MeasurementMinusCFs, FStateDynInputs
