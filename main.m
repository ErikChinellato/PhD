%DEEP KALMAN FILTER MAIN SCRIPT

%******************************************************************************************************
%% START
%MAKE SURE THE LATEST WEIGHTS HAVE BEEN SAVED BEFORE RUNNING AGAIN
%clear;
close all;
clc;

global matlab_have_dictionary
matlab_have_dictionary = 0;

fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEEP KALMAN FILTER~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%*******************************************************************************************************
%% RUNTIME OPTIONS
LoadNetParameters = 0;
InitializeNet = 1; % ~LoadNetParameters;

TrainNet = 1;
TrainingBatchNum = 1000;
TrainingBatchSize = 1;

TestNet = 0;

%*******************************************************************************************************
%% LOADING THE NET PARAMETERS AND INITIALIZING PROPAGATION OPTIONS

%Always save the net parameters using: save('Value of WorkingNetParametersName','NetParameters')
WorkingNetParametersName = 'DefaultNetParameters'; %Edit this to current working file name
 
if LoadNetParameters
    fprintf('********************************************************************************\n');
    fprintf('Loading net parameters contained in: %s\n', WorkingNetParametersName);
    fprintf('********************************************************************************\n\n');
    
    load(WorkingNetParametersName); %Creates variable 'NetParameters'
else
    NetParameters.Experiment = importdata('Experiment.mat');
    NetParameters.Layers = importdata(['LayersExp',NetParameters.Experiment,'.mat']);
    NetParameters.Model = importdata(['ModelExp',NetParameters.Experiment,'.mat']); %Must contain M, K, D (the complete StateDimension-by-Layers matrix!), SamplingTimes (a 1-by-Layers vector), AInit, QInit, RInit and PInit
    NetParameters.C = importdata(['CExp',NetParameters.Experiment,'.mat']);         %Save C in a file 'CExp_.mat'!
    NetParameters.StateDimension = size(NetParameters.C,2);
    NetParameters.ObservationDimension = size(NetParameters.C,1);
    NetParameters.WeightMats = 'Input';                                          %Accepted values: 'Identity', 'Input'
    NetParameters.HiddenDynamicsNumber = 1;
    NetParameters.HiddenDynamicsDimension = importdata(['HiddenDynDimExp',NetParameters.Experiment,'.mat'])*ones(1,1);  %length(NetParameters.HiddenDynamicsDimension) must equal NetParameters.HiddenDynamicsNumber
    NetParameters.DictionaryBlocks = {'Constant','Linear','Quadratic','Cubic'};
    if matlab_have_dictionary
        NetParameters.AllowedDictionaryBlocks = dictionary('Constant', 1, ...
                                                           'Linear', NetParameters.StateDimension, ...
                                                           'Quadratic', NetParameters.StateDimension*(NetParameters.StateDimension+1)/2, ...
                                                           'Cubic', NetParameters.StateDimension*(NetParameters.StateDimension+1)*(NetParameters.StateDimension+2)/6);
        NetParameters.DictionaryDimension = sum(NetParameters.AllowedDictionaryBlocks(NetParameters.DictionaryBlocks));
    else
        NetParameters.AllowedDictionaryBlocks.Constant = 1;
        NetParameters.AllowedDictionaryBlocks.Linear = NetParameters.StateDimension;
        NetParameters.AllowedDictionaryBlocks.Quadratic = NetParameters.StateDimension*(NetParameters.StateDimension+1)/2;
        NetParameters.AllowedDictionaryBlocks.Cubic = NetParameters.StateDimension*(NetParameters.StateDimension+1)*(NetParameters.StateDimension+2)/6;
        NetParameters.DictionaryDimension = 0;
        if any(strcmp(NetParameters.DictionaryBlocks,'Constant'))
            NetParameters.DictionaryDimension = NetParameters.DictionaryDimension + 1;
        end
        if any(strcmp(NetParameters.DictionaryBlocks,'Linear'))
            NetParameters.DictionaryDimension = NetParameters.DictionaryDimension + NetParameters.StateDimension;
        end
        if any(strcmp(NetParameters.DictionaryBlocks,'Quadratic'))
            NetParameters.DictionaryDimension = NetParameters.DictionaryDimension + NetParameters.StateDimension*(NetParameters.StateDimension+1)/2;
        end
        if any(strcmp(NetParameters.DictionaryBlocks,'Cubic'))
            NetParameters.DictionaryDimension = NetParameters.DictionaryDimension + NetParameters.StateDimension*(NetParameters.StateDimension+1)*(NetParameters.StateDimension+2)/6;
        end
    end
    NetParameters.ActivateModelDiscovery = 'Yes';                                  %Accepted values: 'Yes', 'No'
    NetParameters.ModelDiscoveryForceCheck = 1000;   
    NetParameters.ModelDiscoveryUpdateBoth = 'Yes';                                  %Accepted values: 'Yes', 'No'
    NetParameters.ModelDiscoveryMethod = 'OMP';                                     %Accepted values: 'OMP', 'ISTA', 'LH'
    NetParameters.ModelDiscoverySmoothing = 'SGMixed';                                  %Accepted values: 'TV', 'TVMixed', 'TVMixed2', 'SG', 'SGMixed', 'SGMixed2', 'SGMixed3', 'No'
    NetParameters.ModelDiscoveryFirstState = min( 0, floor(NetParameters.Layers/2) );
    [NetParameters.A,NetParameters.D,NetParameters.AtA,NetParameters.B] = ConstructTVMatrices(NetParameters.Layers-NetParameters.ModelDiscoveryFirstState,NetParameters.Model.SamplingTimes);
    NetParameters.WinLen = 31;
    [NetParameters.StencilA0,NetParameters.StencilA1] = ConstructSGMatrices(NetParameters.WinLen);
    NetParameters.ModelDiscoveryRelativeThreshold = 0.5;                            %Only used if NetParameters.ModelDiscoveryMethod = 'OMP' or 'LH'
    NetParameters.ModelDiscoveryStableSupportCondition = 4;
    NetParameters.ModelDiscoveryStableSupportUpdates = 1;
    NetParameters.OMPSparsity = 1;
    NetParameters.ISTAThreshold = 0.1;
    NetParameters.ISTAMaxIt = 100;
    NetParameters.ActivateWhitenessMask = 'Yes';                                    %Accepted values: 'Yes', 'No'
    NetParameters.WhitenessLagCounter = 1;
    NetParameters.WhitenessIterationCheck = 20;
    NetParameters.WhitenessUpdateCheck = 8;
    NetParameters.WhitenessDecreaseThreshold = -1e-3;
    [NetParameters.L,NetParameters.LtL] = ConstructLaplacianMatrices(NetParameters.Layers,NetParameters.Model.SamplingTimes);
    NetParameters.SharedWeights = 'No';                                             %Accepted values: 'Yes', 'No'
    NetParameters.BackPropagation = 'Complete';                                     %Accepted values: 'Truncated', 'Complete'
    NetParameters.ProjectDynamics = 'No';                                           %Accepted values: 'Yes', 'No'
    NetParameters.Jacobians = 'Approximated';                                       %Accepted values: 'Algebraic', 'Approximated'
    NetParameters.FiniteDifferences = 'Central';                                    %Accepted values: 'Central', 'Forward', 'Backward'
    NetParameters.FiniteDifferencesSkip = 1e-9;
    NetParameters.GainLearningRate = (1e-5)/TrainingBatchSize;
    NetParameters.GainLearningRateReduction = 1;
    NetParameters.GainLearningRateIncrease = 1e2;
    NetParameters.DynamicsLearningRate = (1.8e-2)/TrainingBatchSize;
    NetParameters.DynamicsLearningRateReduction = 0.5;
    Pen1Val = 1e0; %mean(diag(NetParameters.Model.invRInit)); %1e2;
    Pen2Val = 1e0*ones(1,NetParameters.Layers);%importdata('WeightsPen.mat');%zeros(1,NetParameters.Layers);%importdata('WeightsPen.mat');
    NormPen = max([Pen1Val,Pen2Val]);
    NetParameters.Penalty0 = 1;%1e0;                          %Penalty for (1/2)*||States{Layers+1}-StateTrue||^2
    NetParameters.Penalty1 = Pen1Val*ones(1,NetParameters.Layers)/NormPen;%1e0/NetParameters.Layers;                              %Penalty for sum_{Layer = 1,...,Layers}(1/2)*||MeasurementMinusCStates{Layer}||^2
    NetParameters.Penalty2 = Pen2Val/NormPen;    %linspace(1,1/NetParameters.Layers,NetParameters.Layers).                  %Penalty for sum_{Layer = 1,...,Layers}(1/2)*||GainMeasurementMinusCFs{Layer}||^2
    NetParameters.Penalty3 = 1e0/(NetParameters.StateDimension*NetParameters.ObservationDimension); %Penalty for (1/2)*||L*TensorizedGains||^2
    NetParameters.Optimizer = 'Adam';                                               %Accepted values: 'Adam', 'SGD'
    NetParameters.BetaMoment1 = 0.9;                                                %Only used if NetParameters.Optimizer = 'Adam'
    NetParameters.BetaMoment2 = 0.999;                                              %Only used if NetParameters.Optimizer = 'Adam'
    NetParameters.Initialization = 'Deterministic';                                 %Accepted values: 'Deterministic', 'DeterministicComplete', 'Random'
    NetParameters.InitializationMean = 0;                                           %Only used if NetParameters.Initialization == 'Random'
    NetParameters.InitializationSigma = 0.0001;                                     %Only used if NetParameters.Initialization == 'Random'
    NetParameters.AdamEpsilon = 1e-16;
    NetParameters.TrainingConditionStop = 'Whiteness';                              %Accepted values: 'Whiteness', 'Residues'
    NetParameters.ResidueDecreaseThreshold = 1e-3;                                 %Only used if NetParameters.TrainingConditionStop == 'Residues'
    
    save(WorkingNetParametersName,'NetParameters');
end

%TO BE REMOVED
if strcmp(NetParameters.Experiment,'1') || strcmp(NetParameters.Experiment,'6')
    KF_InstanceInd = 1;
    MB_KF_V3;
    stateMSE_KF = 1/(size(StatesKF,1)*size(StatesKF,2)) * norm(StatesKF-TrajectoryTrue,'fro')
    stateMSEatTf_KF = 1/(size(StatesKF,1)) * norm(StatesKF(:,end)-StateTrue,'fro')
    mat_MeasurementMinusCStates = cell2mat(MeasurementMinusCStates);
    outMSE_KF = 1/(size(mat_MeasurementMinusCStates,1)*size(mat_MeasurementMinusCStates,2)) * norm(mat_MeasurementMinusCStates,'fro')
    outMSEatTf_KF = 1/(size(mat_MeasurementMinusCStates,1)) * norm(mat_MeasurementMinusCStates(:,end),'fro')
end

%Clear useless allocated variables
clear WorkingNetParametersName

%******************************************************************************************************
%% INITIALIZATION
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name

if InitializeNet 
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Initialize the weights
fprintf('********************************************************************************\n');
fprintf('Initializing weights.\n');
fprintf('********************************************************************************\n\n');

NetWeights = InitializeWeights(NetParameters);

%Save the weights
save(WorkingNetWeightsName,'NetWeights');

fprintf('********************************************************************************\n');
fprintf('Initialization completed. \nInitial weights for the net have been saved in %s and are ready to be used.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear WorkingNetWeightsName NetWeights;
%--------------------------------------------------------------------------
end

%******************************************************************************************************
%% TRAINING
%MAKE SURE TO HAVE A BACKUP COPY OF THE LATEST NET WEIGHTS IN CASE
%SOMETHING BAD HAPPENS DURING TRAINING
%Always save the net weights using: save('Value of WorkingNetWeightsName','NetWeights')
%Always save the training set using: save('Value of WorkingTrainingSetName','TrainingSet')
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name
WorkingTrainingSetName = ['LatestTrainingSetExp',NetParameters.Experiment];

if TrainNet
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Load the training set
load(WorkingTrainingSetName); %Creates variable 'TrainingSet'

%Setup dimensions
TrainInstancesNum = size(TrainingSet,2);

fprintf('********************************************************************************\n');
fprintf('Updating the net weights contained in: %s.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Load the latest net weights
load(WorkingNetWeightsName); %Creates variable 'NetWeights'

%Setup running loss plot and output figure
TrainingResidues = zeros(1,TrainingBatchNum);
PeriodogramResidues = zeros(TrainingBatchNum,2*NetParameters.ObservationDimension,TrainInstancesNum);
%PeriodogramResidues = zeros(TrainingBatchNum,2,TrainInstancesNum);
DynHist = zeros(NetParameters.HiddenDynamicsDimension(NetParameters.HiddenDynamicsNumber),TrainingBatchNum);

figure(1);
set(gcf,'Position',[0 1000 2000 1000]);

%Initialize the moments
[~,Moment1,Moment2] = InitializeGradsAndMoments(NetWeights,NetParameters);
AdamInd = 1;

%Gain Mask set up
GainMask = ones(NetParameters.ObservationDimension,TrainInstancesNum);
LaggedGainMask = ones(NetParameters.ObservationDimension,TrainInstancesNum);
LagCounterInit = NetParameters.WhitenessLagCounter;
LagCounter = LagCounterInit*ones(NetParameters.ObservationDimension,TrainInstancesNum);

%Model discovery set up
ModelDiscoveryUpdates = 0;
ModelDiscoverySupport = logical( true(NetParameters.DictionaryDimension,NetParameters.StateDimension).*(sum(NetParameters.C,1) > 0) );
CurrentSupport = ModelDiscoverySupport;
SupportIsStable = 0;
StableSupportCounter = 0;
StableSupportUpdates = 0;

%Stop conditions set up
InhibitWhitenessCheck = 0;
StopTraining = 0;

%Compute weight matrices
[MeasurementWeightMats,PredictorWeightMats,MeasurementWeightMatsSym,PredictorWeightMatsSym] = ComputeWeightMats(NetParameters);

%Cycle over batch number
for TrainingBatchInd = 1:TrainingBatchNum

    %Check early stop training condition
    if StopTraining
        %Exit outer loop
        break;
    end

    %Reset gradients for new batch but keep the moments intact
    Grads = InitializeGradsAndMoments(NetWeights,NetParameters);

    TrainInstancesPerm = randperm(TrainInstancesNum);
    %Cycle over each training instance in the batch
    for BatchInd = 1:TrainingBatchSize 
        %Randomly select a training instance
        TrainInstanceInd = TrainInstancesPerm(BatchInd);

        %Extract instance
        Inputs = TrainingSet{1,TrainInstanceInd};
        Measurements = TrainingSet{2,TrainInstanceInd};
        FirstState = TrainingSet{3,TrainInstanceInd};
        TrajectoryTrue = TrainingSet{4,TrainInstanceInd};
        StateTrue = TrainingSet{4,TrainInstanceInd}(:,end);
        Dynamic = TrainingSet{5,TrainInstanceInd};
        
        StringProgress = [];
        OverallProgress = fprintf('Training batch number: %d/%d. \nCurrently processing batch instance: %d/%d. \n',TrainingBatchInd,TrainingBatchNum,BatchInd,TrainingBatchSize);
        StringProgress = [StringProgress,OverallProgress];

        PropagationProgress  = fprintf('Propagating instance...\n');
        StringProgress = [StringProgress,PropagationProgress];

        %Propagate input
        [States,MeasurementMinusCStates,GainMeasurementMinusCFs,MeasurementMinusCFs,FStateDynInputs] = PropagateInput(Inputs,Measurements,FirstState,Dynamic,@F,NetWeights,NetParameters);

        %Assemble gains tensor
        TensorizedGains = ConstructTensorizedGains(NetWeights,NetParameters);

        %TO BE REMOVED
        if 0
            StatesDeepKalman = cell2mat(States);
            stateMSE_DeepKalman = 1/(size(StatesDeepKalman,1)*size(StatesDeepKalman,2)) * norm(StatesDeepKalman-TrajectoryTrue,'fro')
            stateMSEatTf_DeepKalman = 1/(size(StatesDeepKalman,1)) * norm(StatesDeepKalman(:,end)-StateTrue,'fro')
            mat_MeasurementMinusCStates = cell2mat(MeasurementMinusCStates);
            outMSE_DeepKalman = 1/(size(mat_MeasurementMinusCStates,1)*size(mat_MeasurementMinusCStates,2)) * norm(mat_MeasurementMinusCStates,'fro')
            outMSEatTf_DeepKalman = 1/(size(mat_MeasurementMinusCStates,1)) * norm(mat_MeasurementMinusCStates(:,end),'fro')
        end

        %Update training residue, cumulative periodograms residues and assemble states evolution
        ShowStates = zeros(NetParameters.Layers+1,NetParameters.StateDimension);
        ShowCorrectorResidues = zeros(NetParameters.Layers,NetParameters.ObservationDimension);
        ShowPredictorResidues = zeros(NetParameters.Layers,NetParameters.ObservationDimension);
        ShowMeasurements = zeros(NetParameters.Layers+1,NetParameters.ObservationDimension); %TO BE REMOVED
        ShowStates(1,:) = FirstState;
        ShowMeasurements(1,:) = Measurements{1}';
        for Layer = 1:NetParameters.Layers
            TrainingResidues(TrainingBatchInd) = TrainingResidues(TrainingBatchInd) + (NetParameters.Penalty1(Layer)/2)*( MeasurementMinusCStates{Layer} )'*MeasurementWeightMats{Layer}*( MeasurementMinusCStates{Layer} )/TrainingBatchSize + (NetParameters.Penalty2(Layer)/2)*( GainMeasurementMinusCFs{Layer} )'*PredictorWeightMats{Layer}*( GainMeasurementMinusCFs{Layer} )/TrainingBatchSize;
            ShowStates(Layer+1,:) = States{Layer+1}';
            ShowMeasurements(Layer+1,:) = Measurements{Layer+1}'; %TO BE REMOVED
            ShowCorrectorResidues(Layer,:) = MeasurementMinusCStates{Layer}';
            ShowPredictorResidues(Layer,:) = MeasurementMinusCFs{Layer}';
        end
        TrainingResidues(TrainingBatchInd) = TrainingResidues(TrainingBatchInd) + (NetParameters.Penalty0/2)*norm( States{end} - StateTrue )^2/TrainingBatchSize + (NetParameters.Penalty3/2)*norm( tensorprod(TensorizedGains,NetParameters.L,3,2), 'fro' )^2/TrainingBatchSize;
        PeriodogramResidues(TrainingBatchInd,:,TrainInstanceInd) = PeriodogramResidues(TrainingBatchInd,:,TrainInstanceInd) + ComputePeriodogramResidue(MeasurementMinusCStates,MeasurementMinusCFs);

        %Check whiteness
        if  strcmp(NetParameters.ActivateWhitenessMask,'Yes')&&( ~InhibitWhitenessCheck )&&( (TrainingBatchInd > NetParameters.WhitenessIterationCheck)&&(AdamInd > NetParameters.WhitenessUpdateCheck) )
            StopCond = ( PeriodogramResidues(TrainingBatchInd,NetParameters.ObservationDimension+1:end,TrainInstanceInd) - PeriodogramResidues(TrainingBatchInd-1,NetParameters.ObservationDimension+1:end,TrainInstanceInd) < NetParameters.WhitenessDecreaseThreshold )';
            %StopCond = ( PeriodogramResidues(TrainingBatchInd,2,TrainInstanceInd) - PeriodogramResidues(TrainingBatchInd-1,2,TrainInstanceInd) < NetParameters.WhitenessDecreaseThreshold )';
            GainMask(:,TrainInstanceInd) = GainMask(:,TrainInstanceInd).*StopCond;
            %GainMask(:,TrainInstanceInd) = GainMask(:,TrainInstanceInd).*StopCond;
            LagCounter = LagCounter - (~GainMask);
            LaggedGainMask(LagCounter == 0) = 0;
        end

        %Check for model discovery update
        UpdateModelDiscovery = strcmp(NetParameters.ActivateModelDiscovery,'Yes')&&( ~InhibitWhitenessCheck )&&( ( AdamInd > NetParameters.ModelDiscoveryForceCheck )||( ~any(sum(LaggedGainMask,2)>0) ) );

        %Decide whether to backpropagate the output
        if ~( UpdateModelDiscovery && strcmp(NetParameters.ModelDiscoveryUpdateBoth,'No') )
            BackPropagationProgress  = fprintf('Back-propagating instance...\n');
            StringProgress = [StringProgress,BackPropagationProgress];

            %Compute jacobians
            [StateJacobians,DynJacobians] = ComputeJacobians(@F,States,NetWeights{end}{Dynamic},Inputs,NetWeights{end}{end},Dynamic,FStateDynInputs,NetParameters);

            %Backpropagate output
            Grads = BackPropagateOutput(StateTrue,Dynamic,States,MeasurementMinusCStates,GainMeasurementMinusCFs,MeasurementMinusCFs,TensorizedGains,MeasurementWeightMatsSym,PredictorWeightMatsSym,Grads,StateJacobians,DynJacobians,NetWeights,NetParameters);
        end

        %Decide whether to update the sparse model discovery matrix
        if UpdateModelDiscovery
            UpdateDiscoveryProgress = fprintf('Updating model discovery matrix...\n');
            StringProgress = [StringProgress,UpdateDiscoveryProgress];

            %Update dynamics
            %NetWeights{end}{end} = UpdateSparseMat(NetWeights,States,Dynamic,NetParameters);
            TempSparseMat = UpdateSparseMat(NetWeights,States,ModelDiscoverySupport,Dynamic,NetParameters);

            %Check support
            if ~SupportIsStable
                NewSupport = (TempSparseMat ~= 0);
                SupportHasNotChanged = ~any( NewSupport - CurrentSupport,'all');

                %Update support
                CurrentSupport = NewSupport;

                if SupportHasNotChanged
                    StableSupportCounter = StableSupportCounter + 1;
                else
                    StableSupportCounter = 0;
                end

                if StableSupportCounter > NetParameters.ModelDiscoveryStableSupportCondition
                    SupportIsStable = 1;
                    ModelDiscoverySupport = CurrentSupport;
                end
            end

            if (StableSupportUpdates < NetParameters.ModelDiscoveryStableSupportUpdates)
                %Update sparse matrix, save hidden parameters and re-initialize Kalman gains
                TempHiddenDynamics = NetWeights{end}{Dynamic};
                NetWeights = InitializeWeights(NetParameters);
                NetWeights{end}{end} = TempSparseMat;
                NetWeights{end}{Dynamic} = TempHiddenDynamics;
    
                %Reset moments since the governing model has changed/is not yet stable
                TempMoment1HiddenDynamics = Moment1{end}{Dynamic};
                TempMoment2HiddenDynamics = Moment2{end}{Dynamic};
                [~,Moment1,Moment2] = InitializeGradsAndMoments(NetWeights,NetParameters);
                Moment1{end}{Dynamic} = TempMoment1HiddenDynamics;
                Moment2{end}{Dynamic} = TempMoment2HiddenDynamics;

                %Slow down the hidden parameters learning after the first model discoveries
                if (ModelDiscoveryUpdates <= 1)
                    NetParameters.DynamicsLearningRate = NetParameters.DynamicsLearningRate*NetParameters.DynamicsLearningRateReduction;
                end

                %Speed up the Kalman gains learning if the support just became stable
                if ( SupportIsStable )&&( StableSupportUpdates == 0 )
                     NetParameters.GainLearningRate = NetParameters.GainLearningRate*NetParameters.GainLearningRateIncrease;
                end
    
                %Reset backpropagation update counter
                AdamInd = 1;
    
                %Reset gain mask
                GainMask = ones(NetParameters.ObservationDimension,TrainInstancesNum);
                LaggedGainMask = ones(NetParameters.ObservationDimension,TrainInstancesNum);
                LagCounter = LagCounterInit*ones(NetParameters.ObservationDimension,TrainInstancesNum);

                %Update stable support update counter
                StableSupportUpdates = StableSupportUpdates + SupportIsStable;

                if strcmp(NetParameters.TrainingConditionStop,'Residues')
                    if StableSupportUpdates == NetParameters.ModelDiscoveryStableSupportUpdates
                        InhibitWhitenessCheck = 1;
                    end
                end
            else
                %Early stop training condition based on whiteness
                fprintf(repmat('\b',1,sum(StringProgress)));
                StopTraining = 1;

                %Exit inner loop
                break;
            end

            ModelDiscoveryUpdates = ModelDiscoveryUpdates + 1;
        end

        fprintf(repmat('\b',1,sum(StringProgress)));
    end

    %Decide whether to update the weights
    if ~( UpdateModelDiscovery && strcmp(NetParameters.ModelDiscoveryUpdateBoth,'No') )
        UpdateWeightsProgress  = fprintf('Updating weights...\n');

        %Update net weights
        [NetWeights,Moment1,Moment2] = UpdateWeights(NetWeights,Grads,Moment1,Moment2,Dynamic,AdamInd,ones(NetParameters.StateDimension,1)-NetParameters.C'*(~sum(LaggedGainMask,2)),NetParameters);
        AdamInd = AdamInd + 1;
        DynHist(:,TrainingBatchInd) = NetWeights{end}{Dynamic};

        fprintf(repmat('\b',1,UpdateWeightsProgress));
    end

    %Save the new weights (we could save the weights only at the end of the 
    %training process, but just in case something bad happens mid-execution... 
    %we do not want to lose hours of training due to a blackout)
    save(WorkingNetWeightsName,'NetWeights'); %This will delete the previously saved weights, make sure to have a backup!

    %Adaptively change the learning rates
    if TrainingResidues(TrainingBatchInd) < NetParameters.GainLearningRate 
        NetParameters.GainLearningRate = NetParameters.GainLearningRate*NetParameters.GainLearningRateReduction;
    end
    if TrainingResidues(TrainingBatchInd) < NetParameters.DynamicsLearningRate 
        NetParameters.DynamicsLearningRate = NetParameters.DynamicsLearningRate*NetParameters.DynamicsLearningRateReduction;
    end

    if strcmp(NetParameters.TrainingConditionStop,'Residues')&&( InhibitWhitenessCheck )&&( AdamInd > NetParameters.WhitenessUpdateCheck )&&( (TrainingResidues(TrainingBatchInd-1)-TrainingResidues(TrainingBatchInd) < NetParameters.ResidueDecreaseThreshold)&&(TrainingResidues(TrainingBatchInd-1)-TrainingResidues(TrainingBatchInd) > 0) )
        %Early stop training condition based on residues decrease
        StopTraining = 1;
    end
    
    %Show training output
    clf;
    subplot(3,3,1); 
    plot(StateTrue,'b+-'); 
    hold on; 
    plot(States{end},'m.-');
    title('Output state comparison');
    xlabel('Nodes');
    legend('True','Estimated','Location','northwest');

    if strcmp(NetParameters.Experiment,'1')
        subplot(3,3,2); 
        plot(NetParameters.C*StateTrue,'b+-'); 
        hold on;
        plot(NetParameters.C*States{end},'m.-');
        title('Output measurements comparison');
        xlabel('Measured nodes');
        legend('True','Estimated');
    end

    if strcmp(NetParameters.Experiment,'2')
        subplot(3,3,2); 
        plot(NetParameters.C*StateTrue,'b+-'); 
        hold on;
        plot(NetParameters.C*States{end},'m.-');
        title('Output measurements comparison');
        xlabel('Measured nodes');
        legend('True','Estimated');
    end

    if strcmp(NetParameters.Experiment,'3')
        subplot(3,3,2); 
        plot(NetParameters.C*StateTrue,'b+-'); 
        hold on;
        plot(NetParameters.C*States{end},'m.-');
        title('Output measurements comparison');
        xlabel('Measured nodes');
        legend('True','Estimated');
    end

    if strcmp(NetParameters.Experiment,'4')
        subplot(3,3,2);
        plot(DynHist(:,1:TrainingBatchInd)','b-o'); 
        hold on;
        yline(-0.4,'m');
        title('Hidden parameters comparison');
        xlabel('Iterate');
        legend('Estimated','True');
    end

    if strcmp(NetParameters.Experiment,'5')
        subplot(3,3,2);
        plot(DynHist(:,1:TrainingBatchInd)','b-o'); 
        hold on;
        yline(-2,'m');
        title('Hidden parameters comparison');
        xlabel('Iterate');
        legend('Estimated','True');
    end

    if strcmp(NetParameters.Experiment,'6')
        subplot(3,3,2);
        plot(squeeze(TensorizedGainsMBKF(1,1,:)),'b-+'); 
        hold on;
        plot(squeeze(TensorizedGains(1,1,:)),'b-o');
        plot(squeeze(TensorizedGainsMBKF(2,2,:)),'r-+');
        plot(squeeze(TensorizedGains(2,2,:)),'r-o');
        title('Kalman Gains entries comparison');
        xlabel('Layer');
        legend('MBKF(1,1)','DKF(1,1)','MBKF(2,2)','DKF(2,2)');
    end

    subplot(3,3,3);
    imagesc(NetWeights{end}{end});
    colorbar;
    title('Reconstructed unmodeled dynamics');

    subplot(3,3,4); 
    semilogy(1:TrainingBatchInd,TrainingResidues(1:TrainingBatchInd),'b-');
    title('Train running loss (average over batch)');
    xlabel('Iterate');
    ylabel('Loss function');

    subplot(3,3,5);
    set(gca, 'ColorOrder', hsv(NetParameters.ObservationDimension), 'NextPlot', 'replacechildren');
    %semilogy(PeriodogramResidues(1:TrainingBatchInd,1));
    semilogy(PeriodogramResidues(1:TrainingBatchInd,1:NetParameters.ObservationDimension));
    legend(strcat('ObservedState:',num2str( (1:NetParameters.ObservationDimension)' )),'Location','southwest');
    title('Running corrector residue periodogram (average over batch)');
    xlabel('Iterate');
    ylabel('Periodogram residues (states)');

    subplot(3,3,6); 
    set(gca, 'ColorOrder', hsv(NetParameters.ObservationDimension), 'NextPlot', 'replacechildren');
    %plot(PeriodogramResidues(1:TrainingBatchInd,NetParameters.ObservationDimension+1:end)-PeriodogramResidues(1:TrainingBatchInd,1:NetParameters.ObservationDimension));
    %semilogy(PeriodogramResidues(1:TrainingBatchInd,2));
    semilogy(PeriodogramResidues(1:TrainingBatchInd,1+NetParameters.ObservationDimension:end));
    legend(strcat('ObservedState:',num2str( (1:NetParameters.ObservationDimension)' )),'Location','southwest');
    title('Running predictor residue periodogram (average over batch)');
    xlabel('Iterate');
    ylabel('Periodogram residues (states)');

    if strcmp(NetParameters.Experiment,'1')
        subplot(3,3,7);
        set(gca, 'ColorOrder', hsv(3*NetParameters.StateDimension), 'NextPlot', 'replacechildren');
        plot([ShowStates*NetParameters.C',StatesKF*NetParameters.C',ShowMeasurements]);
        legend(strcat('EstimatedState:',num2str( [(1:NetParameters.ObservationDimension)';(1:NetParameters.ObservationDimension)';(1:NetParameters.ObservationDimension)'])),'Location','northwest');
        title('States estimates');
        xlabel('Nodes');
    end

    if strcmp(NetParameters.Experiment,'2')
        subplot(3,3,7);
        %set(gca, 'ColorOrder', hsv(NetParameters.StateDimension), 'NextPlot', 'replacechildren');
        %plot(ShowStates);
        %legend(strcat('EstimatedState:',num2str( (1:NetParameters.StateDimension)' )),'Location','northwest');
        %title('States estimates');
        %xlabel('Nodes');

        set(gca, 'ColorOrder', hsv(2*NetParameters.ObservationDimension), 'NextPlot', 'replacechildren');
        plot([ShowStates*NetParameters.C',ShowMeasurements]);
        legend(strcat('EstimatedState:',num2str( [(1:NetParameters.ObservationDimension)';(1:NetParameters.ObservationDimension)'])),'Location','northwest');
        title('States estimates');
        xlabel('Nodes');
    end

    if strcmp(NetParameters.Experiment,'3_')
        subplot(3,3,7);
        set(gca, 'ColorOrder', hsv(NetParameters.StateDimension), 'NextPlot', 'replacechildren');
        plot(ShowStates);
        legend(strcat('EstimatedState:',num2str( (1:NetParameters.StateDimension)' )),'Location','northwest');
        title('States estimates');
        xlabel('Nodes');
    end

    if strcmp(NetParameters.Experiment,'3') 
        subplot(3,3,7);
        set(gca, 'ColorOrder', hsv(2*NetParameters.ObservationDimension), 'NextPlot', 'replacechildren');
        plot([ShowStates*NetParameters.C',ShowMeasurements]);
        legend(strcat('EstimatedState:',num2str( [(1:NetParameters.ObservationDimension)';(1:NetParameters.ObservationDimension)'])),'Location','northwest');
        title('States estimates');
        xlabel('Nodes');
    end

    if strcmp(NetParameters.Experiment,'4') 
        subplot(3,3,7);
        set(gca, 'ColorOrder', hsv(2*NetParameters.StateDimension), 'NextPlot', 'replacechildren');
        plot([ShowStates*NetParameters.C',ShowMeasurements]);
        legend(strcat('EstimatedState:',num2str( [(1:NetParameters.StateDimension)';(1:NetParameters.StateDimension)'])),'Location','northwest');
        title('States estimates');
        xlabel('Nodes');
    end

    if strcmp(NetParameters.Experiment,'5') 
        subplot(3,3,7);
        set(gca, 'ColorOrder', hsv(2*NetParameters.StateDimension), 'NextPlot', 'replacechildren');
        plot([ShowStates*NetParameters.C',ShowMeasurements]);
        legend(strcat('EstimatedState:',num2str( [(1:NetParameters.StateDimension)';(1:NetParameters.StateDimension)'])),'Location','northwest');
        title('States estimates');
        xlabel('Nodes');
    end

    if strcmp(NetParameters.Experiment,'6')
        subplot(3,3,7);
        set(gca, 'ColorOrder', hsv(3*NetParameters.StateDimension), 'NextPlot', 'replacechildren');
        plot([ShowStates*NetParameters.C',StatesKF'*NetParameters.C',ShowMeasurements]);
        legend(strcat('EstimatedState:',num2str( [(1:NetParameters.ObservationDimension)';(1:NetParameters.ObservationDimension)';(1:NetParameters.ObservationDimension)'])),'Location','northwest');
        title('States estimates');
        xlabel('Nodes');
    end

    subplot(3,3,8); 
    set(gca, 'ColorOrder', hsv(NetParameters.StateDimension), 'NextPlot', 'replacechildren');
    semilogy(abs(ShowCorrectorResidues));
    legend(strcat('CorrectorResidues:',num2str( (1:NetParameters.ObservationDimension)' )),'Location','northwest');
    title('Corrector Residues');
    xlabel('Nodes');

    subplot(3,3,9); 
    set(gca, 'ColorOrder', hsv(NetParameters.StateDimension), 'NextPlot', 'replacechildren');
    semilogy(abs(ShowPredictorResidues));
    legend(strcat('PredictorResidues:',num2str( (1:NetParameters.ObservationDimension)' )),'Location','northwest');
    title('Predictor Residues');
    xlabel('Nodes');

    refreshdata; 
    drawnow;
end

fprintf('********************************************************************************\n');
fprintf('Training completed. \nUpdated weights for the net have been saved and are ready to be used.\n');
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear WorkingTrainingSetName WorkingNetWeightsName TrainInstancesNum TrainingResidues PlotTrainingResidues PlotIterations Plot UpdateModelDiscovery Moment1 Moment2 Grads BatchInd TrainInstanceInd StateJacobians DynJacobians Inputs Measurements FirstState StateTrue States MeasurementMinusCStates GainMeasurementMinusCFs MeasurementMinusCFs FStateDynInputs Layer TrainingSet NetWeights LastW AdamInd ShowStates StatesKF ShowMeasurements;
%--------------------------------------------------------------------------
end

%******************************************************************************************************
%% TESTING
%Always save the net weights using: save('Value of WorkingNetWeightsName','NetWeights')
%Always save the testing set using: save('Value of WorkingTestingSetName','LatestTestingSet')
%Always save the testing output using: save('Value of WorkingTestingSetOutputName','LatestTestingSetOutput')
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name
WorkingTestingSetName = 'LatestTestingSet';
WorkingTestingSetOutputName = 'LatestTestingSetOutput';

if TestNet
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');
    
%Load the training set
load(WorkingTestingSetName); %Creates variable 'TrainingSet'
    
%Setup dimensions
TestInstancesNum = size(TestingSet,2);

%Creating a cell of size (6,TestImagesNum+1) to store the testing
%output
TestingSetOutput = cell(2,TestInstancesNum+1);

fprintf('********************************************************************************\n');
fprintf('Loading and using the net weights contained in: %s.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Load the latest net weights
load(WorkingNetWeightsName); %Creates variable 'NetWeights'

%Setup running loss plot
TestingResidues = zeros(1,TestInstancesNum);

PlotTestingResidues = 1;
PlotIterations = 0;

figure(2);
Plot = semilogy(PlotIterations,PlotTestingResidues,'b-');
title('Test running loss');
xlabel('Iterate');
ylabel('Loss function');
Plot.XDataSource = 'PlotIterations';
Plot.YDataSource = 'PlotTestingResidues';

%Compute weight matrices
[MeasurementWeightMats,PredictorWeightMats] = ComputeWeightMats(NetParameters);

%Assemble gains tensor
TensorizedGains = ConstructTensorizedGains(NetWeights,NetParameters);

%Cycle over each testing instance
for TestInstanceInd = 1:TestInstancesNum
    OverallProgress = fprintf('Currently processing testing instance: %d/%d.',TestInstanceInd,TestInstancesNum);

    %Select test instance
    Inputs = TestingSet{1,TestInstanceInd};
    Measurements = TestingSet{2,TestInstanceInd};
    FirstState = TestingSet{3,TestInstanceInd};
    StateTrue = TestingSet{4,TestInstanceInd};
    Dynamic = TrainingSet{5,TestInstanceInd};

    %Propagate input
    [States,MeasurementMinusCStates,GainMeasurementMinusCFs] = PropagateInput(Inputs,Measurements,FirstState,Dynamic,@F,NetWeights,NetParameters);

    %Update testing residue
    for Layer = 1:NetParameters.Layers
        TestingResidues(TestInstanceInd) = TestingResidues(TestInstanceInd) + (NetParameters.Penalty1(Layer)/2)*( MeasurementMinusCStates{Layer} )'*MeasurementWeightMats{Layer}*( MeasurementMinusCStates{Layer} ) + (NetParameters.Penalty2(Layer)/2)*( GainMeasurementMinusCFs{Layer} )'*PredictorWeightMats{Layer}*( GainMeasurementMinusCFs{Layer} );
    end
    TestingResidues(TestInstanceInd) = TestingResidues(TestInstanceInd) + (NetParameters.Penalty0/2)*norm( States{end} - StateTrue )^2 + (NetParameters.Penalty3/2)*norm( tensorprod(TensorizedGains,NetParameters.L,3,2), 'fro' )^2;

    %Plot running loss
    PlotTestingResidues(TestInstanceInd+1) = TestingResidues(TestInstanceInd);
    PlotIterations(TestInstanceInd+1) = TestInstanceInd;
    refreshdata;
    drawnow;
    
    %Store the output
    TestingSetOutput{1,TestInstanceInd} = States;
    TestingSetOutput{2,TestInstanceInd} = MeasurementMinusCStates;

    fprintf(repmat('\b',1,OverallProgress));
end

TestingSetOutput{1,TestInstancesNum+1} = TestingResidues;

%Save the output
save(WorkingTestingSetOutputName,'TestingSetOutput');

fprintf('********************************************************************************\n');
fprintf('Testing completed.\nThe output has been saved in %s and is ready to be analized.\n', WorkingTestingSetOutputName);
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear WorkingNetWeightsName WorkingTestingSetOutputName WorkingTestingSetName TestInstancesNum TestingResidues PlotTestingResidues PlotIterations Plot TestInstanceInd Inputs Measurements FirstState StateTrue States MeasurementMinusCStates GainMeasurementMinusCFs Layer NetWeights TestingSet TestingSetOutput LastW;
%--------------------------------------------------------------------------
end


