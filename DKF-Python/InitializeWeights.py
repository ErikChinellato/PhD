import numpy as np

from DeepKalmanFilter.InitializeSparseDynamicsMat import *
from DeepKalmanFilter.InitializeDynamics import *

def InitializeWeights(NetParameters):
    """
    Initializes the net's weights with Gaussian noise of mean NetParameters.InitializationMean 
    and sigma NetParameters.InitializationSigma.
    """
    # Variables
    Experiment = NetParameters['Experiment']
    Layers = NetParameters['Layers']
    SharedWeights = NetParameters['SharedWeights']
    Initialization = NetParameters['Initialization']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']
    HiddenDynamicsNumber = NetParameters['HiddenDynamicsNumber']
    HiddenDynamicsDimension = NetParameters['HiddenDynamicsDimension']
    DictionaryDimension = NetParameters['DictionaryDimension']

    C = NetParameters['C']
    Model = NetParameters['Model']
    
    if SharedWeights == 'No':
        NetWeights = [None] * (Layers + 1)

        if Initialization == 'Deterministic':
            # Deterministic initialization
            if Experiment == '4':
                P = Model['PInit']
                A = Model['AInit']
                Q = 0. #Model['QInit']
                InvR = Model['invRInit']

                InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
                KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)

                for Layer in range(Layers):
                    NetWeights[Layer] = np.copy(KFGain)

            NetWeights[Layers] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[Layers][Dyn] = InitializeDynamics(HiddenDynamicsDimension[Dyn], Model, Experiment)
            NetWeights[Layers][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)

        if Initialization == 'DeterministicComplete':
            # DeterministcComplete initialization
            if Experiment == '4':
                P = Model['PInit']
                A = Model['AInit']
                Q = Model['QInit']
                InvR = Model['invRInit']

                for Layer in range(Layers):
                    InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
                    P = np.linalg.inv(InvP)
                    KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)
                    NetWeights[Layer] = np.copy(KFGain)

            NetWeights[Layers] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[Layers][Dyn] = InitializeDynamics(HiddenDynamicsDimension[Dyn], Model, Experiment)
            NetWeights[Layers][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)

        if Initialization == 'Random':
            #Random initialization
            Mean = NetParameters['InitializationMean']
            Sigma = NetParameters['InitializationSigma']

            for Layer in range(Layers):
                NetWeights[Layer] = np.random.normal(Mean, Sigma, (ObservationDimension,ObservationDimension))

            NetWeights[Layers] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[Layers][Dyn] = np.random.normal(Mean, Sigma, (HiddenDynamicsDimension[Dyn],1))
            NetWeights[Layers][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)


    if SharedWeights == 'Yes':
        NetWeights = [None] * (2)

        if (Initialization == 'Deterministic') or (Initialization == 'DeterministicComplete'):
            if Experiment == '4':
                P = Model['PInit']
                A = Model['AInit']
                Q = Model['QInit']
                InvR = Model['invRInit']

                InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
                KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)

                NetWeights[0] = np.copy(KFGain)

            NetWeights[1] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[1][Dyn] = InitializeDynamics(HiddenDynamicsDimension[Dyn], Model, Experiment)
            NetWeights[1][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)

        if Initialization == 'Random':
            #Random initialization
            Mean = NetParameters['InitializationMean']
            Sigma = NetParameters['InitializationSigma']

            NetWeights[0] = np.random.normal(Mean, Sigma, (ObservationDimension,ObservationDimension))

            NetWeights[1] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[1][Dyn] = np.random.normal(Mean, Sigma, (HiddenDynamicsDimension[Dyn],1))
            NetWeights[1][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)


    return NetWeights
