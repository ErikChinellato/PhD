%DEEP K-SVD MAIN SCRIPT

%******************************************************************************************************
%% START
%MAKE SURE THE LATEST WEIGHTS HAVE BEEN SAVED BEFORE RUNNING AGAIN
%clear;
close all;
clc;

fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DEEP K-SVD~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%*******************************************************************************************************
%% RUNTIME OPTIONS
LoadNetParameters = 0;
InitializeNet = 0; % ~LoadNetParameters;

TrainNet = 1;
TrainingBatchNum = 360*0.5;
TrainingBatchSize = 5;

TestingImages = [1:100]; %List of numbers ranging 1-500
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
    NetParameters.Layers = 7;
    NetParameters.PatchSize = 8;
    NetParameters.SubImagesSize = 128;
    NetParameters.DictionarySize = 18;
    NetParameters.MLPHiddenSizes = [128, 64, 32]; %[2*NetParameters.PatchSize^2,NetParameters.PatchSize^2,(NetParameters.PatchSize^2)/2]
    NetParameters.MLPLastActivation = 'ReLU';
    NetParameters.SharedWeights = 'No';
    NetParameters.NormalizeDictionary = 'No';
    NetParameters.ProjectLastMLPWeights = 'No';
    NetParameters.ClipPatchDenoised = 'Yes';
    NetParameters.Optimizer = 'Adam';
    NetParameters.BetaMoment1 = 0.2; %Only used if NetParameters.Optimizer = 'Adam'
    NetParameters.BetaMoment2 = 0.299; %Only used if NetParameters.Optimizer = 'Adam'
    NetParameters.LearningRate = (1e-5)/TrainingBatchSize;
    NetParameters.NoiseSigma = 25;
    NetParameters.Epsilon = 1e-16;
    
    save(WorkingNetParametersName,'NetParameters');
end

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
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name

if TrainNet
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRAINING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Working directory
Directory = '/Users/erikchinellato/Documents/MATLAB/PhD/Deep K-SVD/GrayScaleImages/';

%Read train images list
TrainImagesList = readlines([Directory,'train_gray.txt']);

%Setup dimensions
TrainImagesNum = length(TrainImagesList); %All images have the same size of (321,481) or (481,321)...
SubImagesNum = prod( size(imread([Directory,TrainImagesList{1}])) - NetParameters.SubImagesSize + 1 ); %...so this is constant.

fprintf('********************************************************************************\n');
fprintf('Updating the net weights contained in: %s.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Load the latest net weights
load(WorkingNetWeightsName); %Creates variable 'NetWeights'

%Setup image comparison, running loss plot and last dictionary
CleanImageFig = zeros(NetParameters.SubImagesSize);
NoisyImageFig = zeros(NetParameters.SubImagesSize);
DenoisedImageFig = zeros(NetParameters.SubImagesSize);

TrainingResidues = zeros(1,TrainingBatchNum);
PlotTrainingResidues = 1;
PlotIterations = 0;

LastDictionaryFig = zeros(NetParameters.DictionarySize*NetParameters.PatchSize);

figure(1);
set(gcf,'Position',[0 1000 3000 500])

subplot(1,5,1);
ShowCleanImageFig = imshow(CleanImageFig);
title('Clean image');
hold on;

subplot(1,5,2);
ShowNoisyImageFig = imshow(NoisyImageFig);
title('Noisy image');
hold on;

subplot(1,5,3);
ShowDenoisedImageFig = imshow(DenoisedImageFig);
title('Denoised image');
hold on;

subplot(1,5,4);
PlotResidues = plot(PlotIterations,PlotTrainingResidues,'b-');
ylim([1e-3,5e-2]);
yline(1e-2,'r-');
yline(0.0064,'g-');
yline(0.00675,'m-');
yline(0.00605,'m-');
title('Train running loss (average over batch)');
xlabel('Iterate');
ylabel('Mean Squared Error');
PlotResidues.XDataSource = 'PlotIterations';
PlotResidues.YDataSource = 'PlotTrainingResidues';

subplot(1,5,5);
ShowLastDictionaryFig = imshow(LastDictionaryFig);
title('Last dictionary');
hold on;

%Initialize the moments
[~,Moment1,Moment2] = InitializeGradsAndMoments(NetWeights,NetParameters);

%Cycle over batch number
for TrainingBatchInd = 1:TrainingBatchNum
    %Reset gradients for new batch but keep the moments intact
    Grads = InitializeGradsAndMoments(NetWeights,NetParameters);

    %Cycle over each training instance in the batch
    for BatchInd = 1:TrainingBatchSize 
        %Randomly select an image and a subimage within it
        TrainImageInd = randi(TrainImagesNum);
        TrainSubImageInd = randi(SubImagesNum);

        Image = imread([Directory,TrainImagesList{TrainImageInd}]);
        SubImage = double( SelectPatch(Image,NetParameters.SubImagesSize,NetParameters.SubImagesSize,TrainSubImageInd) );
        
        %Add noise
        SubImageNoisy = SubImage + normrnd(0,NetParameters.NoiseSigma,NetParameters.SubImagesSize);

        %Rescale both subimages in [-1,1]
        YNoisy = RescaleImage(SubImageNoisy);
        YClean = RescaleImage(SubImage);
        
        ShowProgress = fprintf('Training batch number: %d/%d. \nCurrently processing batch instance: %d/%d. \n',TrainingBatchInd,TrainingBatchNum,BatchInd,TrainingBatchSize);

        %Propagate input
        [YDenoised,SparseReps,Patches,PatchesDenoised,Lambdas,Mus,Nus,Csis,dSTs,DSparseRepmPatches,XDen] = PropagateInput(YNoisy,NetWeights,NetParameters);
        Lambdas
        %Update running loss
        TrainingResidues(TrainingBatchInd) = TrainingResidues(TrainingBatchInd) + mean( (YDenoised-YClean).^2,'all' )/TrainingBatchSize;
        %Backpropagate output
        Grads = BackpropagateOutput(YClean,YDenoised,SparseReps,Patches,PatchesDenoised,Lambdas,Mus,Nus,Csis,dSTs,DSparseRepmPatches,XDen,Grads,NetWeights,NetParameters);
        fprintf(repmat('\b',1,ShowProgress));
    end

        %Update net weights
        [NetWeights,Moment1,Moment2] = UpdateWeights(NetWeights,Grads,Moment1,Moment2,TrainingBatchInd,NetParameters);
        %Save the new weights (we could save the weights only at the end of the 
        %training process, but just in case something bad happens mid-execution... 
        %we do not want to lose hours of training due to a blackout)
        save(WorkingNetWeightsName,'NetWeights'); %This will delete the previously saved weights, make sure to have a backup!

        %Show last image comparison in batch, plot running loss and display last dictionary
        imshow(mat2gray(YClean),'Parent',ShowCleanImageFig.Parent);
        imshow(mat2gray(YNoisy),'Parent',ShowNoisyImageFig.Parent);
        imshow(mat2gray(YDenoised),'Parent',ShowDenoisedImageFig.Parent);
        PlotTrainingResidues(TrainingBatchInd+1) = TrainingResidues(TrainingBatchInd);
        PlotIterations(TrainingBatchInd+1) = TrainingBatchInd;
        refreshdata(PlotResidues);
        LastDict = normalize(NetWeights{'Dict'}{end},'norm',2);
        for RowInd = 1:NetParameters.DictionarySize
            for ColInd = 1:NetParameters.DictionarySize
                Atom = reshape( LastDict(:,(RowInd-1)*NetParameters.DictionarySize+ColInd) ,NetParameters.PatchSize,NetParameters.PatchSize );
                LastDictionaryFig( (RowInd-1)*NetParameters.PatchSize+1:RowInd*NetParameters.PatchSize, (ColInd-1)*NetParameters.PatchSize+1:ColInd*NetParameters.PatchSize ) = 2*( Atom - min(Atom,[],'all') )/( max(Atom,[],'all') - min(Atom,[],'all') );
            end
        end
        imshow(mat2gray(LastDictionaryFig),'Parent',ShowLastDictionaryFig.Parent);
        drawnow;
end

fprintf('********************************************************************************\n');
fprintf('Training completed. \nUpdated weights for the net have been saved and are ready to be used.\n');
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
%clear WorkingTrainingSetName WorkingNetWeightsName Directory CleanImageFig NoisyImageFig DenoisedImageFig LastDictionaryFig TrainImagesList TrainImagesNum SubImagesNum TrainingResidues PlotTrainingResidues PlotIterations PlotResidues ShowCleanImageFig ShowNoisyImageFig ShowDenoisedImageFig ShowLastDictionaryFig TrainingBatchInd BatchInd TrainImageInd TrainSubImageInd Image SubImage SubImageNoisy NetWeights Moment1 Moment2 YNoisy YClean YDenoised Grads SparseReps Patches PatchesDenoised Lambdas Mus Nus Csis dSTs DSparseRepmPatches XDen UpdatesNum LastDict RowInd ColInd Atom;
%--------------------------------------------------------------------------
end

%******************************************************************************************************
%% TESTING
%Always save the net weights using: save('Value of WorkingNetWeightsName','NetWeights')
%Always save the testing output using: save('Value of WorkingTestingSetOutputName','LatestTestingSetOutput')
WorkingNetWeightsName = 'LatestNetWeights'; %Edit this to current working file name
WorkingTestingSetOutputName = 'LatestTestingSetOutput';

if TestNet
%--------------------------------------------------------------------------
fprintf('********************************************************************************\n');
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('********************************************************************************\n\n');

%Working directory
LoadDirectory = '/Users/erikchinellato/Documents/MATLAB/PhD/Deep K-SVD/GrayScaleImages/';
SaveDirectory = '/Users/erikchinellato/Documents/MATLAB/PhD/Deep K-SVD/DenoisedImages/';
    
%Read testing images list
TestingImagesList = readlines([LoadDirectory,'test_gray.txt']);
    
%Setup dimensions
TestImagesNum = length(TestingImages);

%Creating a cell of size (6,TestImagesNum+1) to store the testing
%output
TestingSetOutput = cell(6,TestImagesNum+1);

fprintf('********************************************************************************\n');
fprintf('Loading and using the net weights contained in: %s.\n', WorkingNetWeightsName);
fprintf('********************************************************************************\n\n');

%Load the latest net weights
load(WorkingNetWeightsName); %Creates variable 'NetWeights'

%Setup running loss plot
TestingResidues = zeros(1,TestImagesNum);

PlotTestingResidues = 1;
PlotIterations = 0;

figure(1);
PlotResidues = plot(PlotIterations,PlotTestingResidues,'b-');
ylim([1e-3,2e-2]);
yline(0.00685,'g-');
title('Test running loss');
xlabel('Iterate');
ylabel('Mean Squared Error');
PlotResidues.XDataSource = 'PlotIterations';
PlotResidues.YDataSource = 'PlotTestingResidues';

%Cycle over each testing image
for TestImageInd = 1:TestImagesNum
    ShowProgress = fprintf('Currently processing testing instance: %d/%d.',TestImageInd,TestImagesNum);

    %Select image
    Image = double( imread([LoadDirectory,TestingImagesList{TestImageInd}]) );
    ImageNoisy = Image + normrnd(0,NetParameters.NoiseSigma,size(Image));
        
    %Rescale in [-1,1]
    YNoisy = RescaleImage(ImageNoisy);
    YClean = RescaleImage(Image);

    %Propagate input
    [YDenoised,SparseReps,Patches,PatchesDenoised] = PropagateInput(YNoisy,NetWeights,NetParameters);
    TestingResidues(TestImageInd) = mean( (YDenoised-YClean).^2,'all' );

    %Plot running loss
    PlotTestingResidues(TestImageInd+1) = TestingResidues(TestImageInd);
    PlotIterations(TestImageInd+1) = TestImageInd;
    refreshdata;
    drawnow;
    
    %Store the output
    TestingSetOutput{1,TestImageInd} = YNoisy;
    TestingSetOutput{2,TestImageInd} = YClean;
    TestingSetOutput{3,TestImageInd} = YDenoised;
    TestingSetOutput{4,TestImageInd} = Patches;
    TestingSetOutput{5,TestImageInd} = PatchesDenoised;
    TestingSetOutput{6,TestImageInd} = SparseReps;
    save([SaveDirectory,TestingImagesList{TestImageInd}(1:end-4)],'YDenoised');

    fprintf(repmat('\b',1,ShowProgress));
end

TestingSetOutput{6,TestImagesNum+1} = TestingResidues;

%Save the output
%save(WorkingTestingSetOutputName,'TestingSetOutput');

fprintf('********************************************************************************\n');
fprintf('Testing completed.\nThe output has been saved in %s and is ready to be analized.\n', WorkingTestingSetOutputName);
fprintf('********************************************************************************\n\n');

%Clear useless allocated variables
clear LoadDirectory SaveDirectory TestingImagesList TestImagesNum TestImageInd WorkingTestingSetOutputName WorkingNetWeightsName NetWeights TestingSetOutput ShowProgress YNoisy YClean YDenoised Patches PatchesDenoised SparseReps TestingResidues Image ImageNoisy PlotResidues PlotTestingResidues PlotIterations;
%--------------------------------------------------------------------------
end


