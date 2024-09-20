%DEEP NMFD MAIN SCRIPT

%******************************************************************************************************
%% START
%MAKE SURE THE LATEST WEIGHTS HAVE BEEN SAVED BEFORE RUNNING AGAIN
clear;
close all;
clc;

fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEEP NMFD~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%*******************************************************************************************************
%% RUNTIME OPTIONS
LoadNetParameters = 0;
InitializeNet = 1;% ~LoadNetParameters;
TrainNet = 1;
TrainingRepsNum = 50;
TrainingBatchSize = 12;
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
    NetParameters.Layers = 25;
    NetParameters.DiscriminativeLayers = 20;
    NetParameters.SparsePenalty = 0;
    NetParameters.Ranks = [7,7,7,7*ones(1,55)];%9*ones(1,58);%[10,10,10,10,10];
    NetParameters.Sources = length(NetParameters.Ranks);
    NetParameters.SourcesOfInterest = [1,2,3];
    NetParameters.NotSourcesOfInterest = setdiff(1:NetParameters.Sources,NetParameters.SourcesOfInterest);
    NetParameters.Initialization = 'NMF';                          %Accepted alues: 'NMF', 'AsIs'
    NetParameters.InitializationBeta = '1';                         %Accepted values: '1', '2'. Only used if NetParameters.Initialization = 'NMF'
    NetParameters.InitializationMaxIt = 5000;                       %Only used if NetParameters.Initialization = 'NMF'
    NetParameters.Objective = 'WienerFilterReconstruction';         %Accepted alues: 'WienerFilterReconstruction', 'WienerFilter'
    NetParameters.ModifyLastNotSourcesOfInterestWeights = 'Yes';    %Accepted values: 'Yes', 'No'
    NetParameters.ReconstructionPenalty = 1e20;                      %Only used if NetParameters.Objective = 'WienerFilterReconstruction'
    NetParameters.Epsilon = 1e-20;
    
    save(WorkingNetParametersName,'NetParameters');
end

%Clear useless allocated variables
clear WorkingNetParametersName;

%******************************************************************************************************
%% INITIALIZATION
%Always save the concatenated clean sources using: save('Value of WorkingConcatenatedCleanSourcesName','ConcatenatedCleanSources')
%Always save the net weights using: save('Value of WorkingNetWeightsName','NetWeights')
WorkingCleanSourcesName = 'LatestCleanSourcesPiano'; %Edit this to current working file name
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name

if InitializeNet 
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Load the clean sources
%BEWARE: at the moment we are assuming the clean sources to be contained in
%a cell array of size (1,NetParameters.Sources), one cell dedicated to each
%source.
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
SizeCS = size(CleanSources{1},1);
for SourceCounter = 1:NetParameters.Sources
    RowNum = size(CleanSources{SourceCounter},1);
    if SizeCS ~= RowNum
        warning('Issue detected at data source %d/%d: number of rows in each clean source matrix must be constant. \nInitialization stopped. \n\n',SourceCounter,NetParameters.Sources);
        return;
    end
end

%Initialize the weights
fprintf('********************************************************************************\n');
fprintf('Initializing weights.\n');
fprintf('********************************************************************************\n\n');

NetWeights = InitializeWeights(CleanSources,NetParameters);

%Save the weights
save(WorkingNetWeightsName,'NetWeights');

fprintf('********************************************************************************\n');
fprintf('Initialization completed. \nInitial weights for the net have been saved in %s and are ready to be used.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear WorkingCleanSourcesName WorkingNetWeightsName CleanSources SizeCS SourceCounter RowNum NetWeights;
%--------------------------------------------------------------------------
end

%******************************************************************************************************
%% TRAINING
%MAKE SURE TO HAVE A BACKUP COPY OF THE LATEST NET WEIGHTS IN CASE
%SOMETHING BAD HAPPENS DURING TRAINING, E.G. YOU USE THE WRONG TRAINING
%SET
%Always save the training set using: save('Value of WorkingTrainingSetName','TrainingSet')
%Always save the net weights using: save('Value of WorkingNetWeightsName','NetWeights')
WorkingTrainingSetName = 'LatestTrainingSetPiano'; %Edit this to current working file name
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
%row for each instance contains a cell array of size (1,2) containing the 
%clean and noise sources for such mixture.
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
    if size(TrainingSet{2,TrainingInstanceCounter},2) ~= 2
        warning('Issue detected at instance %d/%d: number of declared sources in given data not compatible. \nTraining stopped.\n\n',TrainingInstanceCounter,TrainingInstancesNum);
        return;
    end
    
    if size(TrainingSet{1,TrainingInstanceCounter},1) ~= RowNum
        warning('Issue detected at instance %d/%d: number of rows in given data (mixture matrix) not consistent. \nTraining stopped.\n\n',TrainingInstanceCounter,TrainingInstancesNum);
        return;
    end

    for SourceCounter = 1:2
        if size(TrainingSet{2,TrainingInstanceCounter}{SourceCounter},1) ~= RowNum
            warning('Issue detected at instance %d/%d: number of rows in given data (clean source %d/2) not consistent. \nTraining stopped.\n\n',TrainingInstanceCounter,TrainingInstancesNum,SourceCounter);
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
        
        %Reset gradient list for new batch. It is a cell array of size 
        %(2,NetParameters.DiscriminativeLayers) that will contain the gradients
        %for each layer. One cell column is dedicated to each layer; the first 
        %row contains in a cell array of size (1,NetParameters.Sources) the 
        %corresponding layer's negative gradients, while the second row contains
        %in a cell array of size (1,NetParameters.Sources) the corresponding layer's
        %positive gradients
        GradListSum = InitializeGrads(NetParameters,NetWeights);
    
        %Cycle over each training instance in the batch
        for TrainingInstanceCounter = BatchBegin:BatchBegin+TrainingBatchSize-1
            ShowProgress = fprintf('Training repetition: %d/%d. \nCurrently processing training instance: %d/%d. \n',TrainingReps,TrainingRepsNum,TrainingInstanceCounter,TrainingInstancesNum);
    
            %Select instance mixture matrix and clean sources
            MixMat = TrainingSet{1,TrainingInstanceCounter};
            InstanceCleanSources = TrainingSet{2,TrainingInstanceCounter};
            %Propagate input
            HList = PropagateInput(MixMat,NetParameters,NetWeights);
            %Backpropagate output
            GradListSum = BackPropagateOutput(MixMat,HList,InstanceCleanSources,NetWeights,GradListSum,NetParameters);
            fprintf(repmat('\b',1,ShowProgress));
            
            %Update training residue
            InstanceCleanSourceHat = ConstructXHat( NetWeights(NetParameters.SourcesOfInterest,end), HList(NetParameters.SourcesOfInterest,end) );%./ConstructXHat(NetWeights(:,end), HList(:,end) ).*MixMat;
            TrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd) = TrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd) + norm(InstanceCleanSources{1}-InstanceCleanSourceHat)^2/TrainingBatchSize;

            %Plot training reconstruction
            HSourceOfInterest = zeros(1,size(HList{1,1},2)); 
            for i = NetParameters.SourcesOfInterest 
                HSourceOfInterest(i,:) = HList{i,end}; 
            end
            figure(2); 
            imagesc(10*log10(HSourceOfInterest)); 
            colorbar; 
            figure(3); 
            imagesc(10*log10(InstanceCleanSourceHat));
            colorbar;
            figure(4); 
            imagesc(10*log10(InstanceCleanSources{1}));
            colorbar;
            drawnow;
            
        end
      
        %Update net weights
        NetWeights = UpdateWeights(NetWeights,GradListSum,NetParameters,1e1);

        %Plot training loss
        PlotTrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd + 1) = TrainingResidues((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd);
        PlotIterations((TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd + 1) = (TrainingReps-1)*TrainingInstancesNum/TrainingBatchSize + BatchInd;
        refreshdata(PlotResidues);
        drawnow;
        BatchInd = BatchInd + 1;

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
clear -regexp WorkingTrainingSetName WorkingNetWeightsName Perm PermInd TrainingSetOld TrainingSet TrainingInstancesNum RowNum TrainingInstanceCounter TrainingReps BatchBegin GradListSum ShowProgress SourceCounter MixMat InstanceCleanSources HList NetWeights;
%--------------------------------------------------------------------------
end

%******************************************************************************************************
%% TESTING
%Always save the testing set using: save('Value of WorkingTestingSetName','TestingSet')
%Always save the testing set output using: save('Value of WorkingTestingSetOutputName','TestingSetOutput')
WorkingTestingSetName = 'LatestTestingSetPiano'; %Edit this to current working file name
WorkingTestingSetOutputName = 'LatestTestingSetPianoOutput'; %Edit this to current working file name
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

%Selecting the final dictionaries used for the reconstruction
WFinal = NetWeights(:,end);
WFinalSource = WFinal(NetParameters.SourcesOfInterest); %Only the first source is reconstructed

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
    MixMat = TestingSet{TestingInstanceCounter};
    %Propagate input
    HList = PropagateInput(MixMat,NetParameters,NetWeights);
    %Assemble output using the last coefficient matrix HList(:,end)
    TestingSetOutput{1,TestingInstanceCounter} = ConstructXHat( WFinalSource, HList(NetParameters.SourcesOfInterest,end) );
    TestingSetOutput{2,TestingInstanceCounter} = TestingSetOutput{1,TestingInstanceCounter}./( ConstructXHat( WFinal, HList(:,end) ) + eps);
    TestingSetOutput{3,TestingInstanceCounter} = TestingSetOutput{2,TestingInstanceCounter}.*MixMat;
    for HCounter = 1:NetParameters.DiscriminativeLayers+1
        TestingSetOutput{4,TestingInstanceCounter}{HCounter} = HList(:,end-NetParameters.DiscriminativeLayers-1+HCounter);
    end
    fprintf(repmat('\b',1,ShowProgress));
end

%Save the output
save(WorkingTestingSetOutputName,'TestingSetOutput');

fprintf('********************************************************************************\n');
fprintf('Testing completed.\nThe output has been saved in %s and is ready to be analized.\n', WorkingTestingSetOutputName);
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear -regexp WorkingTestingSetName WorkingTestingSetOutputName WorkingNetWeightsName TestingSet NetWeights WFinal WFinalSource TestingInstancesNum TestingSetOutput TestingInstanceCounter ShowProgress MixMat HList HCounter;
%--------------------------------------------------------------------------
end

%******************************************************************************************************




