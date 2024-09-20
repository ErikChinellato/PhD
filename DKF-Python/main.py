import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import copy

from DeepKalmanFilter.Utility import *
from DeepKalmanFilter.ConstructLaplacianMatrices import *
from DeepKalmanFilter.ConstructSGMatrices import *
from DeepKalmanFilter.ConstructTVMatrices import *
from DeepKalmanFilter.BackPropagateOutput import *
from DeepKalmanFilter.ComputeJacobians import *
from DeepKalmanFilter.ComputePeriodogramResidues import *
from DeepKalmanFilter.ComputeWeightMats import *
from DeepKalmanFilter.ConstructTensorizedGains import *
from DeepKalmanFilter.F import *
from DeepKalmanFilter.InitializeGradsAndMoments import *
from DeepKalmanFilter.PropagateInput import *
from DeepKalmanFilter.UpdateSparseMat import *
from DeepKalmanFilter.UpdateWeights import *
from DeepKalmanFilter.InitializeWeights import *

def main(Directory):
    # Print header
    print('********************************************************************************')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEEP KALMAN FILTER~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('********************************************************************************\n')

    # Runtime options
    LoadNetParameters = False
    InitializeNet = True
    TrainNet = True
    TrainingBatchNum = 1000
    TrainingBatchSize = 1

    # Loading the net parameters and initializing propagation options
    WorkingNetParametersName = 'DefaultNetParameters.mat'

    if LoadNetParameters:
        print('********************************************************************************')
        print(f'Loading net parameters contained in: {WorkingNetParametersName}')
        print('********************************************************************************\n')
        
        # Load net parameters
        NetParameters = sio.loadmat(WorkingNetParametersName,squeeze_me = True)['NetParameters']
    else:
        NetParameters = {}
        NetParameters['Experiment'] = sio.loadmat(Directory+'Experiment.mat',squeeze_me = True)['Experiment']
        NetParameters['Layers'] = sio.loadmat(Directory+f'LayersExp{NetParameters["Experiment"]}.mat',squeeze_me = True)['Layers']
        NetParameters['Model'] = loadPickle(Directory+f'ModelExp{NetParameters["Experiment"]}.mat')
        NetParameters['C'] = sio.loadmat(Directory+f'CExp{NetParameters["Experiment"]}.mat',squeeze_me = True)['C']
        NetParameters['StateDimension'] = NetParameters['C'].shape[1]
        NetParameters['ObservationDimension'] = NetParameters['C'].shape[0]
        NetParameters['WeightMats'] = 'Input'                                                   #Supported values: Input, Identity  
        NetParameters['HiddenDynamicsNumber'] = 1
        NetParameters['HiddenDynamicsDimension'] = [sio.loadmat(Directory+f'HiddenDynDimExp{NetParameters["Experiment"]}.mat',squeeze_me = True)['HiddenDynDim']]*NetParameters['HiddenDynamicsNumber']
        NetParameters['DictionaryBlocks'] = ['Constant', 'Linear', 'Quadratic', 'Cubic']
        NetParameters['AllowedDictionaryBlocks'] = {
            'Constant': 1,
            'Linear': NetParameters['StateDimension'],
            'Quadratic': NetParameters['StateDimension'] * (NetParameters['StateDimension'] + 1) // 2,
            'Cubic': NetParameters['StateDimension'] * (NetParameters['StateDimension'] + 1) * (NetParameters['StateDimension'] + 2) // 6
        }
        NetParameters['DictionaryDimension'] = sum(NetParameters['AllowedDictionaryBlocks'][block] for block in NetParameters['DictionaryBlocks'])
        NetParameters['ActivateModelDiscovery'] = 'Yes'
        NetParameters['ModelDiscoveryForceCheck'] = 1000
        NetParameters['ModelDiscoveryUpdateBoth'] = 'Yes'
        NetParameters['ModelDiscoveryMethod'] = 'OMP'                                            #Supported values: OMP, LH  
        NetParameters['ModelDiscoverySmoothing'] = 'SGMixed2'                                    #Supported values: TV, TVMixed, SG, SGMixed1, SGMixed2
        NetParameters['ModelDiscoveryFirstState'] = min(0, NetParameters['Layers'] // 2)
        NetParameters['A'], NetParameters['D'], NetParameters['AtA'], NetParameters['B'] = ConstructTVMatrices( NetParameters['Layers'] - NetParameters['ModelDiscoveryFirstState'], NetParameters['Model']['SamplingTimes'] )
        NetParameters['WinLen'] = 31
        NetParameters['StencilA0'], NetParameters['StencilA1'] = ConstructSGMatrices(NetParameters['WinLen'])
        NetParameters['ModelDiscoveryRelativeThreshold'] = 0.8
        NetParameters['ModelDiscoveryStblSuppCondition'] = 4
        NetParameters['ModelDiscoveryStblSuppUpdates'] = 1
        NetParameters['OMPSparsity'] = 1
        NetParameters['ActivateWhitenessMask'] = 'Yes'
        NetParameters['WhitenessLagCounter'] = 1
        NetParameters['WhitenessIterationCheck'] = 20
        NetParameters['WhitenessUpdateCheck'] = 8
        NetParameters['WhitenessDecreaseThreshold'] = -1e-3
        NetParameters['L'], NetParameters['LtL'] = ConstructLaplacianMatrices( NetParameters['Layers'], NetParameters['Model']['SamplingTimes'] )
        NetParameters['SharedWeights'] = 'No'
        NetParameters['BackPropagation'] = 'Complete'                                           #Supported values: Complete, Truncated  
        NetParameters['ProjectDynamics'] = 'No'
        NetParameters['Jacobians'] = 'Approximated'                                             #Supported values: Approximated, Algebraic  
        NetParameters['FiniteDifferences'] = 'Central'                                          #Supported values: Forward, Backward, Central  
        NetParameters['FiniteDifferencesSkip'] = 1e-9
        NetParameters['GainLearningRate'] = (1e-5) / TrainingBatchSize
        NetParameters['GainLearningRateReduction'] = 1
        NetParameters['GainLearningRateIncrease'] = 1e2
        NetParameters['DynamicsLearningRate'] = (1.8e-2) / TrainingBatchSize
        NetParameters['DynamicsLearningRateReduction'] = 0.2
        Pen1Val = 1e0
        Pen2Val = np.ones(NetParameters['Layers']) * 1e0
        NormPen = max(Pen1Val, Pen2Val.max())
        NetParameters['Penalty0'] = 1
        NetParameters['Penalty1'] = np.ones(NetParameters['Layers']) * Pen1Val / NormPen
        NetParameters['Penalty2'] = Pen2Val / NormPen
        NetParameters['Penalty3'] = 1e0 / (NetParameters['StateDimension'] * NetParameters['ObservationDimension'])
        NetParameters['Optimizer'] = 'Adam'
        NetParameters['BetaMoment1'] = 0.9
        NetParameters['BetaMoment2'] = 0.999
        NetParameters['Initialization'] = 'Deterministic'                                       #Supported values: Random, Deterministic, DeterministcComplete  
        NetParameters['InitializationMean'] = 0
        NetParameters['InitializationSigma'] = 0.0001
        NetParameters['AdamEpsilon'] = 1e-16
        NetParameters['TrainingConditionStop'] = 'Residues'                                    #Supported values: Whiteness, Residues  
        NetParameters['ResidueDecreaseThreshold'] = 1e-3
        
        # Save net parameters
        sio.savemat(Directory+WorkingNetParametersName, {'NetParameters': NetParameters})

    # Clear variables
    del WorkingNetParametersName

    # Initialization
    WorkingNetWeightsName = 'LatestNetWeights.mat'

    if InitializeNet:
        print('********************************************************************************')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('********************************************************************************\n')

        # Initialize weights
        print('********************************************************************************')
        print('Initializing weights.')
        print('********************************************************************************\n')

        # Define function `InitializeWeights`
        NetWeights = InitializeWeights(NetParameters)

        # Save weights
        savePickle(Directory+WorkingNetWeightsName,NetWeights)

        print('********************************************************************************')
        print(f'Initialization completed. Initial weights for the net have been saved in {WorkingNetWeightsName} and are ready to be used.')
        print('********************************************************************************\n')


    # MAKE SURE TO HAVE A BACKUP COPY OF THE LATEST NET WEIGHTS IN CASE SOMETHING BAD HAPPENS DURING TRAINING
    WorkingNetWeightsName = 'LatestNetWeights.mat'  # Edit this to current working file name
    WorkingTrainingSetName = f'LatestTrainingSetExp{NetParameters["Experiment"]}.mat'

    if TrainNet:
        #--------------------------------------------------------------------------
        print('********************************************************************************')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('********************************************************************************\n')

        # Load the training set
        TrainingSet = loadPickle(Directory+WorkingTrainingSetName)

        # Setup dimensions
        TrainInstancesNum = np.shape(TrainingSet[0])[0]

        print('********************************************************************************')
        print(f'Updating the net weights contained in: {WorkingNetWeightsName}.')
        print('********************************************************************************\n')

        # Load the latest net weights
        NetWeights = loadPickle(Directory+WorkingNetWeightsName)

        # Setup running loss plot and output figure
        TrainingResidues = np.zeros((TrainingBatchNum,1))
        PeriodogramResidues = np.zeros((TrainingBatchNum, 2 * NetParameters['ObservationDimension'], TrainInstancesNum))
        DynHist = np.zeros((NetParameters['HiddenDynamicsDimension'][NetParameters['HiddenDynamicsNumber']-1], TrainingBatchNum)) #THIS NEEDS TO BE FIXED

        # Set up the figure
        plt.figure(1)
        plt.gcf().set_size_inches(20, 10)

        # Initialize the moments
        Moment1, Moment2 = InitializeGradsAndMoments(NetWeights, NetParameters)[1:]

        AdamInd = 1

        # Gain Mask set up
        GainMask = np.ones((NetParameters['ObservationDimension'], TrainInstancesNum))
        LaggedGainMask = np.ones((NetParameters['ObservationDimension'], TrainInstancesNum))
        LagCounterInit = NetParameters['WhitenessLagCounter']
        LagCounter = LagCounterInit * np.ones((NetParameters['ObservationDimension'], TrainInstancesNum))

        # Model discovery set up
        ModelDiscoveryUpdates = 0
        ModelDiscoverySupport = (np.ones((NetParameters['DictionaryDimension'],NetParameters['StateDimension']))*np.sum(NetParameters['C'], axis=0) > 0).astype(bool)
        CurrentSupport = ModelDiscoverySupport
        SupportIsStable = 0
        StableSupportCounter = 0
        StableSupportUpdates = 0

        # Stop conditions set up
        InhibitWhitenessCheck = 0
        StopTraining = 0

        # Compute weight matrices
        MeasurementWeightMats, PredictorWeightMats, MeasurementWeightMatsSym, PredictorWeightMatsSym = ComputeWeightMats(Directory, NetParameters)

        # Cycle over batch number
        for TrainingBatchInd in range(1, TrainingBatchNum+1):

            # Check early stop training condition
            if StopTraining:
                # Exit outer loop
                break

            # Reset gradients for new batch but keep the moments intact
            Grads = InitializeGradsAndMoments(NetWeights, NetParameters)[0]

            TrainInstancesPerm = np.random.permutation(TrainInstancesNum)
            
            # Cycle over each training instance in the batch
            for BatchInd in range(1, TrainingBatchSize+1):
                # Randomly select a training instance
                TrainInstanceInd = TrainInstancesPerm[BatchInd-1]

                # Extract instance
                Inputs = TrainingSet[0][TrainInstanceInd]
                Measurements = TrainingSet[1][TrainInstanceInd]
                FirstState = TrainingSet[2][TrainInstanceInd]
                TrajectoryTrue = TrainingSet[3][TrainInstanceInd]
                StateTrue = TrainingSet[4][TrainInstanceInd]
                Dynamic = TrainingSet[5][TrainInstanceInd]

                # Print progress
                #OverallProgress = f'Training batch number: {TrainingBatchInd}/{TrainingBatchNum}. \nCurrently processing batch instance: {BatchInd}/{TrainingBatchSize}. \n'
                #print(OverallProgress, end='')

                #PropagationProgress = 'Propagating instance...\n'
                #print(PropagationProgress, end='')

                # Propagate input
                States, MeasurementMinusCStates, GainMeasurementMinusCFs, MeasurementMinusCFs, FStateDynInputs = PropagateInput(Inputs, Measurements, FirstState, Dynamic, F, NetWeights, NetParameters)

                # Assemble gains tensor
                TensorizedGains = ConstructTensorizedGains(NetWeights, NetParameters)

                # Update training residue, cumulative periodograms residues and assemble states evolution
                ShowStates = np.zeros((NetParameters['Layers']+1, NetParameters['StateDimension']))
                ShowCorrectorResidues = np.zeros((NetParameters['Layers'], NetParameters['ObservationDimension']))
                ShowPredictorResidues = np.zeros((NetParameters['Layers'], NetParameters['ObservationDimension']))
                ShowMeasurements = np.zeros((NetParameters['Layers']+1, NetParameters['ObservationDimension']))  # TO BE REMOVED
                ShowStates[0:1,:] = FirstState.T
                ShowMeasurements[0:1,:] = Measurements[:,0:1].T

                for Layer in range(1, NetParameters['Layers'] + 1):
                    TrainingResidues[TrainingBatchInd-1] += np.squeeze( (NetParameters['Penalty1'][Layer-1]/2)*( MeasurementMinusCStates[Layer-1].T )@MeasurementWeightMats[Layer-1]@( MeasurementMinusCStates[Layer-1] )/TrainingBatchSize + (NetParameters['Penalty2'][Layer-1]/2)*( GainMeasurementMinusCFs[Layer-1].T )@PredictorWeightMats[Layer-1]@( GainMeasurementMinusCFs[Layer-1] )/TrainingBatchSize )
                    ShowStates[Layer,:] = States[Layer].T
                    ShowMeasurements[Layer,:] = Measurements[:,Layer:Layer+1].T  # TO BE REMOVED
                    ShowCorrectorResidues[Layer-1,:] = MeasurementMinusCStates[Layer-1].T
                    ShowPredictorResidues[Layer-1,:] = GainMeasurementMinusCFs[Layer-1].T

                TrainingResidues[TrainingBatchInd-1] += np.squeeze( (NetParameters['Penalty0']/2)*np.linalg.norm( States[-1] - StateTrue )**2/TrainingBatchSize + (NetParameters['Penalty3']/2)*np.linalg.norm( np.tensordot(TensorizedGains, NetParameters['L'], axes=([2], [1])) )**2/TrainingBatchSize )
                PeriodogramResidues[TrainingBatchInd-1,:,TrainInstanceInd-1] += ComputePeriodogramResidue(MeasurementMinusCStates, MeasurementMinusCFs)

                # Check whiteness
                if ( (NetParameters['ActivateWhitenessMask'] == 'Yes') and (not InhibitWhitenessCheck) and (TrainingBatchInd > NetParameters['WhitenessIterationCheck'] and AdamInd > NetParameters['WhitenessUpdateCheck']) ):
                    StopCond = (PeriodogramResidues[TrainingBatchInd-1,NetParameters['ObservationDimension']:,TrainInstanceInd-1] - PeriodogramResidues[TrainingBatchInd-2,NetParameters['ObservationDimension']:,TrainInstanceInd-1] < NetParameters['WhitenessDecreaseThreshold']).T
                    GainMask[:,TrainInstanceInd-1] *= StopCond
                    LagCounter -= ( np.logical_not(GainMask)*1 )
                    LaggedGainMask[LagCounter == 0] = 0

                # Check for model discovery update
                UpdateModelDiscovery = ( (NetParameters['ActivateModelDiscovery'] == 'Yes') and (not InhibitWhitenessCheck) and ( (AdamInd > NetParameters['ModelDiscoveryForceCheck']) or (not np.any(np.sum(LaggedGainMask, axis=1) > 0)) ) )

                # Decide whether to backpropagate the output
                if not ( UpdateModelDiscovery and (NetParameters['ModelDiscoveryUpdateBoth'] == 'No') ):
                    #BackPropagationProgress = 'Back-propagating instance...\n'
                    #print(BackPropagationProgress, end='')

                    # Compute jacobians
                    StateJacobians, DynJacobians = ComputeJacobians(F, States, NetWeights[-1][Dynamic-1], Inputs, NetWeights[-1][-1], Dynamic, FStateDynInputs, NetParameters)

                    # Backpropagate output
                    Grads = BackPropagateOutput(StateTrue, Dynamic, States, MeasurementMinusCStates, GainMeasurementMinusCFs, MeasurementMinusCFs, TensorizedGains, MeasurementWeightMatsSym, PredictorWeightMatsSym, Grads, StateJacobians, DynJacobians, NetWeights, NetParameters)

                # Decide whether to update the sparse model discovery matrix
                if UpdateModelDiscovery:
                    #UpdateDiscoveryProgress = 'Updating model discovery matrix...\n'
                    #print(UpdateDiscoveryProgress, end='')

                    # Update dynamics
                    TempSparseMat = UpdateSparseMat(NetWeights, States, ModelDiscoverySupport, Dynamic, NetParameters)

                    # Check support
                    if not SupportIsStable:
                        NewSupport = (TempSparseMat != 0)
                        SupportHasNotChanged = not np.any(NewSupport != CurrentSupport)

                        # Update support
                        CurrentSupport = NewSupport

                        if SupportHasNotChanged:
                            StableSupportCounter += 1
                        else:
                            StableSupportCounter = 0

                        if StableSupportCounter > NetParameters['ModelDiscoveryStblSuppCondition']:
                            SupportIsStable = 1
                            ModelDiscoverySupport = CurrentSupport

                    if StableSupportUpdates < NetParameters['ModelDiscoveryStblSuppUpdates']:
                        # Update sparse matrix, save hidden parameters and re-initialize Kalman gains
                        TempHiddenDynamics = NetWeights[-1][Dynamic-1]
                        NetWeights = InitializeWeights(NetParameters)
                        NetWeights[-1][-1] = copy.deepcopy(TempSparseMat)
                        NetWeights[-1][Dynamic-1] = copy.deepcopy(TempHiddenDynamics)

                        # Reset moments since the governing model has changed/is not yet stable
                        TempMoment1HiddenDynamics = Moment1[-1][Dynamic-1]
                        TempMoment2HiddenDynamics = Moment2[-1][Dynamic-1]
                        Moment1, Moment2 = InitializeGradsAndMoments(NetWeights, NetParameters)[1:]
                        Moment1[-1][Dynamic-1] = copy.deepcopy(TempMoment1HiddenDynamics)
                        Moment2[-1][Dynamic-1] = copy.deepcopy(TempMoment2HiddenDynamics)

                        # Slow down the hidden parameters learning after the first model discoveries
                        if ModelDiscoveryUpdates <= 1:
                            NetParameters['DynamicsLearningRate'] *= NetParameters['DynamicsLearningRateReduction']

                        # Speed up the Kalman gains learning if the support just became stable
                        if SupportIsStable and StableSupportUpdates == 0:
                            NetParameters['GainLearningRate'] *= NetParameters['GainLearningRateIncrease']

                        # Reset backpropagation update counter
                        AdamInd = 1

                        # Reset gain mask
                        GainMask = np.ones((NetParameters['ObservationDimension'], TrainInstancesNum))
                        LaggedGainMask = np.ones((NetParameters['ObservationDimension'], TrainInstancesNum))
                        LagCounter = LagCounterInit * np.ones((NetParameters['ObservationDimension'], TrainInstancesNum))

                        # Update stable support update counter
                        StableSupportUpdates += int(SupportIsStable)

                        if NetParameters['TrainingConditionStop'] == 'Residues':
                            if StableSupportUpdates == NetParameters['ModelDiscoveryStblSuppUpdates']:
                                InhibitWhitenessCheck = 1
                    else:
                        # Early stop training condition based on whiteness
                        #print('\b' * len(StringProgress), end='')
                        StopTraining = 1

                        # Exit inner loop
                        break

                    ModelDiscoveryUpdates += 1

            # Decide whether to update the weights
            if not ( UpdateModelDiscovery and (NetParameters['ModelDiscoveryUpdateBoth'] == 'No') ):
                #UpdateWeightsProgress = 'Updating weights...\n'
                #print(UpdateWeightsProgress, end='')

                # Update net weights
                NetWeights, Moment1, Moment2 = UpdateWeights(NetWeights, copy.deepcopy(Grads), Moment1, Moment2, Dynamic, AdamInd, np.ones((NetParameters['StateDimension'], 1)) - NetParameters['C'].T@( np.logical_not(np.sum(LaggedGainMask, axis=1))*1 ), NetParameters)

                AdamInd += 1
                DynHist[:,TrainingBatchInd-1] = NetWeights[-1][Dynamic-1]

                #print('\b' * len(UpdateWeightsProgress), end='')

            # Save the new weights
            # This will delete the previously saved weights, make sure to have a backup!
            savePickle(Directory+WorkingNetWeightsName,NetWeights)

            # Adaptively change the learning rates
            if TrainingResidues[TrainingBatchInd-1] < NetParameters['GainLearningRate']:
                NetParameters['GainLearningRate'] *= NetParameters['GainLearningRateReduction']
            if TrainingResidues[TrainingBatchInd-1] < NetParameters['DynamicsLearningRate']:
                NetParameters['DynamicsLearningRate'] *= NetParameters['DynamicsLearningRateReduction']

            if ( (NetParameters['TrainingConditionStop'] == 'Residues') and InhibitWhitenessCheck and (AdamInd > NetParameters['WhitenessUpdateCheck']) and ( (TrainingResidues[TrainingBatchInd-2] - TrainingResidues[TrainingBatchInd-1] < NetParameters['ResidueDecreaseThreshold']) and (TrainingResidues[TrainingBatchInd-2] - TrainingResidues[TrainingBatchInd-1] > 0) ) ):
                # Early stop training condition based on residues decrease
                StopTraining = 1
            
            # Show training output
            plt.figure(1, figsize=(20, 10))
            plt.clf()

            plt.subplot(3, 3, 1)
            plt.plot(StateTrue, 'b+-')
            plt.plot(States[-1], 'm.-')
            plt.title('Output state comparison')
            plt.xlabel('Nodes')
            plt.legend(['True','Estimated'],loc='upper right')

            if NetParameters['Experiment'] == '4':
                plt.subplot(3, 3, 2)
                plt.plot(DynHist[0,:TrainingBatchInd], 'b+-')
                plt.axhline(y=-0.4, color='m', linestyle='-')
                plt.title('Hidden parameters comparison')
                plt.xlabel('Iterate')
                plt.legend(['Estimated','True'])

            plt.subplot(3, 3, 3)
            plt.imshow(NetWeights[-1][-1], aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title('Reconstructed unmodeled dynamics')

            plt.subplot(3, 3, 4)
            plt.semilogy(range(1, TrainingBatchInd + 1), TrainingResidues[:TrainingBatchInd], 'b-')
            plt.title('Train running loss (average over batch)')
            plt.xlabel('Iterate')
            plt.ylabel('Loss function')

            plt.subplot(3, 3, 5)
            plt.gca().set_prop_cycle(color=plt.cm.hsv(np.linspace(0, 1, NetParameters['ObservationDimension']+1)))
            plt.semilogy(range(1, TrainingBatchInd + 1), PeriodogramResidues[:TrainingBatchInd, :NetParameters['ObservationDimension'],0])
            plt.legend([f'ObservedState:{i+1}' for i in range(NetParameters['ObservationDimension'])], loc='lower left')
            plt.title('Running corrector residue periodogram (average over batch)')
            plt.xlabel('Iterate')
            plt.ylabel('Periodogram residues (states)')

            plt.subplot(3, 3, 6)
            plt.gca().set_prop_cycle(color=plt.cm.hsv(np.linspace(0, 1, NetParameters['ObservationDimension']+1)))
            plt.semilogy(PeriodogramResidues[:TrainingBatchInd, NetParameters['ObservationDimension']:,0])
            plt.legend([f'ObservedState:{i+1}' for i in range(NetParameters['ObservationDimension'])], loc='lower left')
            plt.title('Running predictor residue periodogram (average over batch)')
            plt.xlabel('Iterate')
            plt.ylabel('Periodogram residues (states)')

            if NetParameters['Experiment'] == '4':
                plt.subplot(3, 3, 7)
                plt.gca().set_prop_cycle(color=plt.cm.hsv(np.linspace(0, 1, 2 * NetParameters['StateDimension']+1)))
                plt.plot(np.hstack((ShowStates @ NetParameters['C'].T, ShowMeasurements)))
                plt.legend([f'EstimatedState:{i+1}' for i in range(NetParameters['ObservationDimension'])] * 2, loc='upper left')
                plt.title('States estimates')
                plt.xlabel('Nodes')

            plt.subplot(3, 3, 8)
            plt.gca().set_prop_cycle(color=plt.cm.hsv(np.linspace(0, 1, NetParameters['StateDimension']+1)))
            plt.semilogy(np.abs(ShowCorrectorResidues))
            plt.legend([f'CorrectorResidues:{i+1}' for i in range(NetParameters['ObservationDimension'])], loc='upper left')
            plt.title('Corrector Residues')
            plt.xlabel('Nodes')

            plt.subplot(3, 3, 9)
            plt.gca().set_prop_cycle(color=plt.cm.hsv(np.linspace(0, 1, NetParameters['StateDimension']+1)))
            plt.semilogy(np.abs(ShowPredictorResidues))
            plt.legend([f'PredictorResidues:{i+1}' for i in range(NetParameters['ObservationDimension'])], loc='upper left')
            plt.title('Predictor Residues')
            plt.xlabel('Nodes')

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

    print('********************************************************************************')
    print('Training completed.')
    print('Updated weights for the net have been saved and are ready to be used.')
    print('********************************************************************************\n')


                        