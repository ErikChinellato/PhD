%DEEP NMF MAIN SCRIPT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLES:
%
% NetParameters: *MANDATORY TO BE INITIALIZED OR LOADED* struct with fields
%   - Layers: 
%       Number of total layers in the net.
%       Default value = 25
%   - DiscriminativeLayers: 
%       Number of discriminative layers in the net. They are the last layers. 
%       Clearly DiscriminativeLayers <= Layers.
%       Default value = 4
%   - SparsePenalty: 
%       Penalty for the sparsity regularizer on the H factors.
%       Default value = 5
%   - ContextFrames: 
%       Number of (context) frames concatenated for the first
%       non-discriminative layers.
%       Default value = 9
%   - Sources: 
%       Number of sources we are modelling
%       Default value = 2
%   - Ranks: 
%       Vector containing the number of basis functions to allocate to
%       each source.
%       Default value = [100, 100]
%   - Initialization:
%       Type of initialization for the net's weights. Accepted values: 
%       'Discriminative', 'NonDiscriminative'.
%       Default value = 'NonDiscriminative'.
%   - WeightsInitialization:
%       Type of initialization structure for the net's weights. Only has an 
%       effect if NetParameters.Initialization == 'Discriminative'. Accepted values: 
%       'CausalityPreserving', 'NoCausalityPreserving'.
%       Default value = 'NoCausalityPreserving'.
%   - SourceWeights: 
%       Weights for the reconstruction error of each source.
%       NOT YET SUPPORTED VALUES DIFFERENT FROM DEFAULT, HARDCODED AS IS.
%       Default value = [1, 0] (we are reconstructing only the first source)
%   - BetaInner: 
%       Value for the beta divergence in the inner layers of the
%       net (all layers except the reconstruction step at the output).
%       NOT YET SUPPORTED VALUES DIFFERENT FROM DEFAULT, HARDCODED AS IS.
%       Default value = 1 
%   - BetaOuter: 
%       Value for the beta divergence in the outer layer of the
%       net (last reconstruction step).
%       NOT YET SUPPORTED VALUES DIFFERENT FROM DEFAULT, HARDCODED AS IS.
%       Default value = 2
%   - Objective:
%       Type of objective to add at the last layer (reconstruction layer).
%       Accepted values: 'WienerFilter', 'WeightedWienerFilter', 'SNRMask',
%       'WienerFilterReconstruction'
%       Default value = 'WienerFilter'
%   - DiscriminativePropagation:
%       Type of propagation mode in the discriminative layers. Only has an
%       effect if NetParameters.ContextFrames > 1.
%       Accepted values: 'Context', 'NoContext'
%       Default value = 'NoContext'
%   - ForwardAndBackPropagation:
%       Type of backpropagation algorithm in the discriminative layers. Only
%       has an effect if NetParameters.DiscriminativePropagation == 'Context'.
%       Accepted values: 'CausalityPreserving', 'NoCausalityPreserving'
%       Default value = 'NoCausalityPreserving'
%   - ForwardPropagationProjection:
%       Type of projection to apply in the discriminative layers to the
%       coefficients' matrices. Only has an effect if
%       NetParameters.ForwardAndBackPropagation == 'CausalityPreserving'.
%       Accepted values: 'NoEnergyPreserving', 'EnergyPreserving'
%       Default value = 'NoEnergyPreserving'
% PropagationOptions: *OPTIONAL* struct with fields
%   - Epsilon: 
%       Small positive value for thresholding.
%       Default value = 2^-52
%   - HInput:
%       Initial matrix for the input propagation.
%       Default value = max(Epsilon,rand(sum(Ranks),size(X,2)))
%   - MaxIt:
%       Maximum number of iterations during initializations of the weights.
%       It is the same for all sources.
%       Default value = 2000;
%   - HInit:
%       Cell array of initial matrices (one for each source) for the weight
%       initialization. (ContextCCS = ContextConcatenatedCleanSources)
%       IF Initialization == 'NonDiscriminative':
%       Default value = { max(Epsilon,rand(Ranks(1),size(ContextConcatenatedCleanSources{1},2))), 
%                           max(Epsilon,rand(Ranks(2),Size(ContextConcatenatedCleanSources{2},2))) }
%       IF Initialization == 'Discriminative':
%       Default value = { { max(epsilon, rand(InitializationRanks{1}(SourceInstance),SizeCS{2}{1}(SourceInstance))), ... }, 
%                          { max(epsilon, rand(InitializationRanks{2}(SourceInstance),SizeCS{2}{2}(SourceInstance))), ... } }
%   - WInit:
%       Cell array of initial matrices (one for each source) for the weight
%       initialization. (ContextCCS = ContextConcatenatedCleanSources)
%       IF Initialization == 'NonDiscriminative':
%       Default value = { max(Epsilon, ContextCCS{1}(:,randperm(size(ContextCCS,2),Ranks(1)))),
%                           max(Epsilon, ContextCCS{2}(:,randperm(size(ContextCCS,2),Ranks(2)))) }
%       IF Initialization = 'Discriminative':
%       Default value = { { max(epsilon, ContextCS{1}{SourceInstance}(:,randperm(SizeCS{2}{1}(SourceInstance),InitializationRanks{1}(SourceInstance)))), ... },
%                          { max(epsilon, ContextCS{2}{SourceInstance}(:,randperm(SizeCS{2}{2}(SourceInstance),InitializationRanks{2}(SourceInstance)))), ... }
%   -InitializationRanks:
%       Cell array of size (1,Sources) where each element is a vector
%       containing the factorization ranks to allocate to each training 
%       clean source IF Initialization == 'Discriminative'. Clearly
%       sum(InitializationRanks{i}) = Ranks(i).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%******************************************************************************************************
%% START
%MAKE SURE THE LATEST WEIGHTS HAVE BEEN SAVED BEFORE RUNNING AGAIN
clear;
close all;
clc;

fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEEP NMF~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%*******************************************************************************************************
%% RUNTIME OPTIONS
LoadNetParameters = 0;
InitializeNet = 1;% ~LoadNetParameters;
TrainNet = 1;
TrainingRepsNum = 3;
TrainingBatchSize = 1;
TestNet = 1;

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
    NetParameters.Layers = 500;
    NetParameters.DiscriminativeLayers = 4;
    NetParameters.SparsePenalty = 0;
    NetParameters.ContextFrames = 7;
    NetParameters.Sources = 2;
    NetParameters.Ranks = [8*10, 7*1];
    NetParameters.Initialization = 'NonDiscriminative';
    NetParameters.WeightsInitialization = 'CausalityPreserving';
    NetParameters.DiscriminativePropagation = 'Context';
    NetParameters.ForwardAndBackPropagation = 'NoCausalityPreserving';
    NetParameters.ForwardPropagationProjection = 'NoEnergyPreserving';
    %NetParameters.SourceWeights = [1, 0]; %Hardcoded
    %NetParameters.BetaInner = 1; %Hardcoded
    %NetParameters.BetaOuter = 2; %Hardcoded
    NetParameters.Objective = 'WienerFilter';
    
    save(WorkingNetParametersName,'NetParameters');
end

%Initializing propagation options (optional)
PropagationOptions = struct;
PropagationOptions.MaxIt = 5000;
PropagationOptions.Epsilon = 10^-16;%2^-52;E
%PropagationOptions.Epsilon = single(10^-16);%2^-52; %REMOVE SINGLE
%PropagationOptions.HInput = single(importdata('H.mat')); %REMOVE THIS
PropagationOptions.InitializationRanks = {8*ones(1,10), 7*ones(1,1)};%{12*ones(1,3), 12*ones(1,55)}; %{22*ones(1,3),[23*ones(1,3),20]};%{8*ones(1,10),7};%{8*ones(1,19),7};% {[8,8,8,8,8],7};%{6*ones(1,15),5};%{10*ones(1,15),5};%{[1,1,3,3,3,1,2,3,3,3,1,4,4,4,3],5};

%Clear useless allocated variables
clear -regexp WorkingNetParametersName;

%******************************************************************************************************
%% INITIALIZATION
%Always save the concatenated clean sources using: save('Value of WorkingConcatenatedCleanSourcesName','ConcatenatedCleanSources')
%Always save the net weights using: save('Value of WorkingNetWeightsName','NetWeights')
WorkingCleanSourcesName = 'LatestCleanSourcesSpindle'; %Edit this to current working file name
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name

if InitializeNet 
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Load the clean sources
%BEWARE: at the moment we are assuming the clean sources to be contained in
%a cell array of size (1,NetParameters.Sources), one cell dedicated to each
%source. Moreover, each element is a cell array containing the clean source 
%instances for the corresponding source.
fprintf('********************************************************************************\n');
fprintf('Loading clean sources contained in: %s.\n', WorkingCleanSourcesName);
fprintf('********************************************************************************\n\n');

load(WorkingCleanSourcesName); %Creates variable 'CleanSources'

%Make sure that NetParameters contains a compatible number of sources.
if size(CleanSources,2) ~= NetParameters.Sources
   warning('Issue detected: number of declared sources not compatible with given data. \nInitialization stopped.\n\n');
   return;
end

%Make sure that each clean source matrix has the same number of rows
%and set up the sizes
SizeCS = cell(1,2);
[SizeCS{1},~] = size(CleanSources{1}{1});
for SourceCounter = 1:NetParameters.Sources
    SourcesNum = size(CleanSources{SourceCounter},2);
    for SourceInstance = 1:SourcesNum
        [RowNum,SizeCS{2}{SourceCounter}(SourceInstance)] = size(CleanSources{SourceCounter}{SourceInstance});
        if SizeCS{1} ~= RowNum
            warning('Issue detected at data source %d/%d and instance %d/%d: number of rows in each clean source matrix must be constant. \nInitialization stopped. \n\n',SourceCounter,NetParameters.Sources,SourceInstance,SourcesNum);
            return;
        end
    end
end

%Initialize the weights
fprintf('********************************************************************************\n');
fprintf('Initializing weights.\n');
fprintf('********************************************************************************\n\n');

[NetWeights,HFinal] = InitializeWeights(CleanSources,SizeCS,PropagationOptions,NetParameters);

%Save the weights
save(WorkingNetWeightsName,'NetWeights');

fprintf('********************************************************************************\n');
fprintf('Initialization completed. \nInitial weights for the net have been saved in %s and are ready to be used.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear -regexp WorkingCleanSourcesName WorkingNetWeightsName CleanSources SizeCS SourceCounter SourcesNum SourceInstance RowNum NetWeights;
%--------------------------------------------------------------------------
end

%******************************************************************************************************
%% TRAINING
%MAKE SURE TO HAVE A BACKUP COPY OF THE LATEST NET WEIGHTS IN CASE
%SOMETHING BAD HAPPENS DURING TRAINING, E.G. YOU USE THE WRONG TRAINING
%SET
%Always save the training set using: save('Value of WorkingTrainingSetName','TrainingSet')
%Always save the net weights using: save('Value of WorkingNetWeightsName','NetWeights')
WorkingTrainingSetName = 'LatestTrainingSetSpindle'; %Edit this to current working file name
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name

if TrainNet
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Load the training set
%BEWARE: at the moment we are assuming the training set to be a cell array
%of size (2,TrainingInstancesNum), one cell column dedicated to each training instance. 
%In particular, the first row for each instance contains the mixture matrix and the second 
%row for each instance contains a cell array of size (1,NetParameters.Sources)
%containing the clean sources for such mixture.
fprintf('********************************************************************************\n');
fprintf('Loading training set contained in: %s.\n', WorkingTrainingSetName);
fprintf('********************************************************************************\n\n');

load(WorkingTrainingSetName); %Creates variable 'TrainingSet'

%Detecting the number of instances
TrainingInstancesNum = size(TrainingSet,2);

RowNum = size(TrainingSet{1,1},1); %This quantity must be constant and consistent with the sources
%Make sure that NetParameters contains a compatible number of sources and
%that dimensions are compatible throughout the training set
for TrainingInstanceCounter = 1:TrainingInstancesNum
    if size(TrainingSet{2,TrainingInstanceCounter},2) ~= NetParameters.Sources
        warning('Issue detected at instance %d/%d: number of declared sources not compatible with given data. \nTraining stopped.\n\n',TrainingInstanceCounter,TrainingInstancesNum);
        return;
    end
    
    if size(TrainingSet{1,TrainingInstanceCounter},1) ~= RowNum
        warning('Issue detected at instance %d/%d: number of rows in given data (mixture matrix) not consistent. \nTraining stopped.\n\n',TrainingInstanceCounter,TrainingInstancesNum);
        return;
    end

    for SourceCounter = 1:NetParameters.Sources
        if size(TrainingSet{2,TrainingInstanceCounter}{SourceCounter},1) ~= RowNum
            warning('Issue detected at instance %d/%d: number of rows in given data (clean source %d/%d) not consistent. \nTraining stopped.\n\n',TrainingInstanceCounter,TrainingInstancesNum,SourceCounter,NetParameters.Sources);
            return;
        end
    end
end

fprintf('********************************************************************************\n');
fprintf('Updating the net weights contained in: %s.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Set up training loss plot
TrainingResidues = zeros(1,TrainingRepsNum*TrainingInstancesNum/TrainingBatchSize);
PlotTrainingResidues = 1;
PlotIterations = 0;

figure(1);
PlotResidues = semilogy(PlotIterations,PlotTrainingResidues,'b-');
title('Train running loss (average over batch)');
xlabel('Iterate');
ylabel('Squared Error');
PlotResidues.XDataSource = 'PlotIterations';
PlotResidues.YDataSource = 'PlotTrainingResidues';

%Cell array of size (2,NetParameters.DiscriminativeLayers) that will
%contain the gradients for each layer. One cell column is dedicated
%to each layer; the first row contains the corresponding layer's
%negative gradient, while the second row the positive gradient.
GradListSum = cell(2,NetParameters.DiscriminativeLayers);

%Cycle over epochs
for TrainingReps = 1:TrainingRepsNum
    %Permute training instances
    Perm = randperm(TrainingInstancesNum);
    TrainingSetOld = TrainingSet;
    for PermInd = 1:TrainingInstancesNum
        TrainingSet{1,Perm(PermInd)} = TrainingSetOld{1,PermInd};
        TrainingSet{2,Perm(PermInd)} = TrainingSetOld{2,PermInd};
    end

    %Set up batch counter
    BatchInd = 1;
    
    %Cycle over batches
    for BatchBegin = 1:TrainingBatchSize:TrainingInstancesNum
        %Load the latest net weights
        load(WorkingNetWeightsName); %Creates variable 'NetWeights'
        
        %Reset gradient list for new batch
        WeightsSize = size(NetWeights{end});
        for LayerNum = 1:NetParameters.DiscriminativeLayers
           GradListSum{1,LayerNum} = zeros(WeightsSize); 
           GradListSum{2,LayerNum} = zeros(WeightsSize); 
        end
    
        %Cycle over each training instance in the batch
        for TrainingInstanceCounter = BatchBegin:BatchBegin+TrainingBatchSize-1
            ShowProgress = fprintf('Training repetition: %d/%d. \nCurrently processing training instance: %d/%d. \n',TrainingReps,TrainingRepsNum,TrainingInstanceCounter,TrainingInstancesNum);
    
            %Select instance mixture matrix and clean sources
            MixMat = TrainingSet{1,TrainingInstanceCounter};
            InstanceCleanSources = TrainingSet{2,TrainingInstanceCounter};
            %Propagate input
            HList = PropagateInput(MixMat,PropagationOptions,NetParameters,NetWeights);
            %Backpropagate output
            GradListSum = BackPropagateOutput(MixMat,HList,InstanceCleanSources,PropagationOptions,NetParameters,NetWeights,GradListSum);

            %Update training residue
            InstanceCleanSourceHat = NetWeights{end}(end-size(TrainingSet{1,1},1)+1:end,1:NetParameters.Ranks(1))*HList{end}(1:NetParameters.Ranks(1),:); %MixMat.*( NetWeights{end}(:,1:NetParameters.Ranks(1))*HList{end}(1:NetParameters.Ranks(1),:)./( NetWeights{end}*HList{end} )  );
            TrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd) = TrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd) + norm(InstanceCleanSources{1}-InstanceCleanSourceHat)^2/TrainingBatchSize;


            fprintf(repmat('\b',1,ShowProgress));
        end

        %Plot training loss
        PlotTrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd + 1) = TrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd);
        PlotIterations((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd + 1) = (TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd;
        refreshdata(PlotResidues);
        drawnow;
        BatchInd = BatchInd + 1;
        
        %Update net weights
        NetWeights = UpdateWeights(PropagationOptions,NetParameters,NetWeights,GradListSum);
        %Save the new weights (we could save the weights only at the end of the 
        %training process, but just in case something bad happens mid-execution... 
        %we do not want to lose hours of training due to a blackout)
        save(WorkingNetWeightsName,'NetWeights'); %This will delete the previously saved weights, make sure to have a backup!
    end
end

fprintf('********************************************************************************\n');
fprintf('Training completed. \nUpdated weights for the net have been saved and are ready to be used.\n');
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear -regexp WorkingTrainingSetName WorkingNetWeightsName Perm PermInd TrainingSetOld TrainingSet TrainingInstancesNum RowNum TrainingInstanceCounter TrainingReps BatchBegin WeightsSize %GradListSum ShowProgress SourceCounter LayerNum MixMat InstanceCleanSources HList NetWeights;
%--------------------------------------------------------------------------
end

%******************************************************************************************************
%% TESTING
%Always save the testing set using: save('Value of WorkingTestingSetName','TestingSet')
%Always save the testing set output using: save('Value of WorkingTestingSetOutputName','TestingSetOutput')
WorkingTestingSetName = 'LatestTestingSetSpindle'; %Edit this to current working file name
WorkingTestingSetOutputName = 'LatestTestingSetSpindleOutput'; %Edit this to current working file name
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name

if TestNet
%-------------------------------------------------------------------------- 
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Load the testing set 
%BEWARE: at the moment we are assuming the testing set to be a cell array
%of size (1,TestingInstancesNum), one cell dedicated to each testing instance. 
%Testing instances are mixture matrices.
fprintf('********************************************************************************\n');
fprintf('Loading testing set contained in: %s.\n', WorkingTestingSetName);
fprintf('********************************************************************************\n\n');

load(WorkingTestingSetName); %Creates variable 'TestingSet'

fprintf('********************************************************************************\n');
fprintf('Loading and using the net weights contained in: %s.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

load(WorkingNetWeightsName); %Creates variable 'NetWeights'

%REMOVE THIS
%for Layer = 1:4
%    NetWeights{Layer} = single(NetWeights{Layer});
%end

%Selecting the final dictionaries used for the reconstruction
RowNum = size(TestingSet{1},1);
WFinal = NetWeights{end}(end-RowNum+1:end,:);
WFinalSource = WFinal(end-RowNum+1:end,1:NetParameters.Ranks(1)); %Only the first source is reconstructed

%Detecting the number of instances
TestingInstancesNum = size(TestingSet,2);

%Creating a cell array of size (4,TestingInstancesNum) to contain the output,  
%one cell column dedicated to each output instance. In particular, the first row
%for each instance contains the reconstructed clean source (spectrogram),
%the second row for each instance contains the weights for an STFT weiner filter
%separating the clean source, the third row for each instance contains the
%reconstructed and 'wiener filtered' clean source (spectrogram) and the fourth row
%contains a cell with the complete history of H matrices.
TestingSetOutput = cell(4,TestingInstancesNum);

%Cycle over each testing instance
for TestingInstanceCounter = 1:TestingInstancesNum
    ShowProgress = fprintf('Currently processing testing instance: %d/%d.',TestingInstanceCounter,TestingInstancesNum);
    
    %Select instance mixture matrix
    %REMOVE SINGLE
    %MixMat = single(TestingSet{TestingInstanceCounter});
    MixMat = TestingSet{TestingInstanceCounter};
    %Propagate input
    HList = PropagateInput(MixMat,PropagationOptions,NetParameters,NetWeights);
    %Assemble output using the last coefficient matrix HList{end}
    TestingSetOutput{1,TestingInstanceCounter} = WFinalSource*HList{end}(1:NetParameters.Ranks(1),:);
    TestingSetOutput{2,TestingInstanceCounter} = TestingSetOutput{1,TestingInstanceCounter}./(WFinal*HList{end}+eps);
    TestingSetOutput{3,TestingInstanceCounter} = TestingSetOutput{2,TestingInstanceCounter}.*MixMat;
    for HCounter = 1:NetParameters.DiscriminativeLayers+1
        TestingSetOutput{4,TestingInstanceCounter}{HCounter} = HList{end-NetParameters.DiscriminativeLayers-1+HCounter};
    end
    fprintf(repmat('\b',1,ShowProgress));
end

%Save the output
save(WorkingTestingSetOutputName,'TestingSetOutput');

fprintf('********************************************************************************\n');
fprintf('Testing completed.\nThe output has been saved in %s and is ready to be analized.\n', WorkingTestingSetOutputName);
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear -regexp WorkingTestingSetName WorkingTestingSetOutputName WorkingNetWeightsName TestingSet NetWeights WFinal WFinalSource RowNum TestingInstancesNum TestingSetOutput TestingInstanceCounter ShowProgress MixMat HList HCounter;
%--------------------------------------------------------------------------
end

%******************************************************************************************************




