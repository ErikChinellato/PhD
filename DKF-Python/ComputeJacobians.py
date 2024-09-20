import numpy as np

from DeepKalmanFilter.DynJacobian import *
from DeepKalmanFilter.StateJacobian import *

def ComputeJacobians(F, States, Dyn, Inputs, SparseMat, Dynamic, FStateDynInputs, NetParameters):
    """
    Computes the Jacobians of F at the different layers of the net. StateJacobians & DynJacobians are lists of size (1, NetParameters['Layers']) where
    StateJacobians[0] = [] since it is not used during backpropagation.
    """
    # Variables
    Experiment = NetParameters['Experiment']
    Layers = NetParameters['Layers']
    Jacobians = NetParameters['Jacobians']
    N = NetParameters['StateDimension']
    P = NetParameters['HiddenDynamicsDimension']

    # Setup output
    StateJacobians = [None] * Layers
    DynJacobians = [None] * Layers

    if Jacobians == 'Approximated':
        # Approximate Jacobians with finite differences
        if Experiment == '4':
            for Layer in range(1, Layers):
                StateJacobians[Layer] = StateJacobian(F, States[Layer], Dyn, Inputs[:,Layer:Layer+1], SparseMat, FStateDynInputs[Layer], Layer, N, NetParameters)
            
            for Layer in range(Layers):
                DynJacobians[Layer] = DynJacobian(F, States[Layer], Dyn, Inputs[:,Layer:Layer+1], SparseMat, FStateDynInputs[Layer], Layer, N, P[Dynamic-1], NetParameters)
    
    elif Jacobians == 'Algebraic':
        # Set Jacobians to their exact algebraic representation, when possible
        if Experiment == '4':
            for Layer in range(1, Layers):
                # Uncomment and define StateJacobianAlgebraic function when available
                # StateJacobians[Layer] = StateJacobianAlgebraic(F, States[Layer], Dyn, Inputs[Layer], SparseMat, FStateDynInputs[Layer], Layer, N, NetParameters)
                pass
            
            for Layer in range(Layers):
                # Uncomment and define DynJacobianAlgebraic function when available
                # DynJacobians[Layer] = DynJacobianAlgebraic(F, States[Layer], Dyn, Inputs[Layer], SparseMat, FStateDynInputs[Layer], Layer, N, P[Dynamic], NetParameters)
                pass
    
    return StateJacobians, DynJacobians
