function [InitialWeights,HFinal] = InitializeWeights(CleanSources,SizeCS,PropagationOptions,NetParameters)
%INITIALIZEWEIGHTS: initializes the weights in the net to the W matrix
%resulting from applying sparse NMF to the (context and concatenated) clean sources.
%ConcatenatedCleanSources is a cell array of size (1,NetParameters.Sources),
%containing in each cell the concatenation of all clean (as in, not mixed) 
%sources in the available training set. SizeCS is a cell of size (1,2) where
%the first element is a scalar containing the (constant) number of rows of 
%each element in CleanSources and the second is a cell array of size
%(1,NetParameters.Sources). Each element is a vector with as many elements as clean sources
%(of the corresponding type), in which and each element contains the number of columns
%of the selected clean source.

%% VARIABLES INITIALIZATION
%Variables
C = NetParameters.DiscriminativeLayers;
S = NetParameters.Sources;
T = NetParameters.ContextFrames;
SparsePen = NetParameters.SparsePenalty;
FactRanks = NetParameters.Ranks;
Initialization = NetParameters.Initialization;
DiscriminativePropagation = NetParameters.DiscriminativePropagation;
WeightsInitialization = NetParameters.WeightsInitialization;

CumSumR = [0,cumsum(FactRanks)];
R = CumSumR(end); %sum(FactRanks);

m = SizeCS{1};

%Propagation options for initialization
if ~isfield(PropagationOptions,'MaxIt')
    maxIt = 500;
else
    maxIt = PropagationOptions.MaxIt; 
end
if ~isfield(PropagationOptions,'Epsilon')
    epsilon = 2^-52;
else
    epsilon = PropagationOptions.Epsilon;
end

%% NONDISCRIMINATIVE INITIALIZATION
if strcmp(Initialization, 'NonDiscriminative')
%Constructing the concatenated clean sources
ConcatenatedCleanSources = cell(1,S);
ColNumCCS = zeros(1,S);
for SourceCounter = 1:S
    for SourceInstance = 1:size(CleanSources{SourceCounter},2)
        ConcatenatedCleanSources{SourceCounter} = [ConcatenatedCleanSources{SourceCounter},CleanSources{SourceCounter}{SourceInstance}];
    end
    ColNumCCS(SourceCounter) = size(ConcatenatedCleanSources{SourceCounter},2); %sum(SizeCS{2}{SourceCounter})
end

%Constructing the context version of the concatenated clean sources
ContextCCS = cell(1,S);
for SourceCounter = 1:S
    ContextCCS{SourceCounter} = ConstructContextMat(ConcatenatedCleanSources{SourceCounter},m,ColNumCCS(SourceCounter),T); 
end

%Initialization options for WInit and HInit
if ~isfield(PropagationOptions,'HInit')
   HInitList  = cell(1,S);
   for SourceCounter = 1:S
       HInitList{SourceCounter} = max(epsilon, rand(FactRanks(SourceCounter),ColNumCCS(SourceCounter)));
   end
else
   HInitList  = PropagationOptions.HInit; 
end
if ~isfield(PropagationOptions,'WInit')
   WInitList = cell(1,S);
   for SourceCounter = 1:S
       WInitList{SourceCounter} = max(epsilon, ContextCCS{SourceCounter}(:,randperm(ColNumCCS(SourceCounter),FactRanks(SourceCounter))));
   end
else
   WInitList = PropagationOptions.WInit;
end

%Dictionary update for each source
WFinal = zeros(m*T,R);
HFinal = cell(1,1); %HERE

for SourceCounter = 1:S
    X = ContextCCS{SourceCounter};
    W = WInitList{SourceCounter};
    H = HInitList{SourceCounter};
    for ItCounter = 1:maxIt
        %Update H
        H = UpdateInitH(X,W,H,SparsePen,epsilon);
        %Update W
        W = UpdateInitW(X,W,H,epsilon);
        %Normalize W and H
        ColumnMax = max(W);
        W = max(epsilon,W./ColumnMax);
        H = max(epsilon,H.*ColumnMax');
    end
    
    WFinal(:,(1+CumSumR(SourceCounter)):CumSumR(SourceCounter+1)) = W;
    
    if SourceCounter == 1 %HERE
        HFinal{1} = H;
    end
end

%Weights initialization
InitialWeights = cell(1,C+1);
InitialWeights{1} = WFinal;

if strcmp(DiscriminativePropagation, 'NoContext')
    WFinalDiscriminative = WFinal(end-m+1:end,:)./max(WFinal(end-m+1:end,:)); %No context, normalize
end

if strcmp(DiscriminativePropagation, 'Context')
    WFinalDiscriminative = WFinal; %Context, already normalized
end

for DiscriminativeLayerCounter = 2:C+1
    InitialWeights{DiscriminativeLayerCounter} = WFinalDiscriminative;
end 
end

%% DISCRIMINATIVE INITIALIZATION
if strcmp(Initialization, 'Discriminative')
%Setting up the cumulative initialization ranks
InitRanks = PropagationOptions.InitializationRanks;
CumSumIR = cell(1,S);
LastInd = 0;
for SourceCounter = 1:S
    CumSumIR{SourceCounter} = [LastInd,LastInd+cumsum(InitRanks{SourceCounter})];
    LastInd = CumSumIR{SourceCounter}(end);
end  

%Constructing the context version of the clean sources
ContextCS = cell(1,S);
SourcesNum = zeros(1,S);
for SourceCounter = 1:S
    SourcesNum(SourceCounter) = size(CleanSources{SourceCounter},2);
    for SourceInstance = 1:SourcesNum(SourceCounter)
        ContextCS{SourceCounter}{SourceInstance} = ConstructContextMat(CleanSources{SourceCounter}{SourceInstance},m,SizeCS{2}{SourceCounter}(SourceInstance),T); 
    end
end

%Initialization options for WInit and HInit
if ~isfield(PropagationOptions,'HInit')
   HInitList = cell(1,S);
   for SourceCounter = 1:S
       for SourceInstance = 1:SourcesNum(SourceCounter)
           HInitList{SourceCounter}{SourceInstance} = max(epsilon, rand(InitRanks{SourceCounter}(SourceInstance),SizeCS{2}{SourceCounter}(SourceInstance)));
       end
   end
else
   HInitList  = PropagationOptions.HInit; 
end
if ~isfield(PropagationOptions,'WInit')
   WInitList = cell(1,S);
   for SourceCounter = 1:S
       for SourceInstance = 1:SourcesNum(SourceCounter)
           WInitList{SourceCounter}{SourceInstance} = max(epsilon, ContextCS{SourceCounter}{SourceInstance}(:,1:InitRanks{SourceCounter}(SourceInstance)));
                                                      %max(epsilon, ContextCS{SourceCounter}{SourceInstance}(:,randperm(SizeCS{2}{SourceCounter}(SourceInstance),InitRanks{SourceCounter}(SourceInstance)))); 
                                                      %max(epsilon, rand(size(ContextCS{SourceCounter}{SourceInstance},1),InitRanks{SourceCounter}(SourceInstance)));
       end
   end
else
   WInitList = PropagationOptions.WInit;
end

%Dictionary update for each source
WFinal = zeros(m*T,R);
HFinal = cell(1,15); %HERE

for SourceCounter = 1:S
    for SourceInstance = 1:SourcesNum(SourceCounter)
        X = ContextCS{SourceCounter}{SourceInstance};
        W = WInitList{SourceCounter}{SourceInstance};
        H = HInitList{SourceCounter}{SourceInstance};
        
        if strcmp(WeightsInitialization, 'NoCausalityPreserving')
            %Here we do not preserve the hankel-like structure of the initialized 
            %discriminative weights.
            MaskH = ones(InitRanks{SourceCounter}(SourceInstance),SizeCS{2}{SourceCounter}(SourceInstance));
            MaskW = ones(m*T,InitRanks{SourceCounter}(SourceInstance));
        end

        if strcmp(WeightsInitialization, 'CausalityPreserving')
            %Here we preserve the hankel-like structure of the initialized 
            %discriminative weights.
            MaskH = ones(InitRanks{SourceCounter}(SourceInstance),SizeCS{2}{SourceCounter}(SourceInstance));
            for RowInd = 1:InitRanks{SourceCounter}(SourceInstance)-1
                MaskH(RowInd,RowInd+1:SizeCS{2}{SourceCounter}(SourceInstance)) = 0; 
                MaskH(RowInd,1:RowInd-1) = 0;%HERE
            end
            MaskH(RowInd+1,1:RowInd) = 0;%HERE

            MaskW = ones(m*T,InitRanks{SourceCounter}(SourceInstance));
            for BlockInd = 1:T-1     
                MaskW(m*(BlockInd-1)+1:m*BlockInd,1:T-BlockInd) = 0; 
            end 
        end
        
        for ItCounter = 1:maxIt
            %Update H
            H = UpdateInitH(X,W,H,SparsePen,epsilon);
            %Update W
            W = UpdateInitW(X,W,H,epsilon);
            %Normalize W and H
            ColumnMax = max(W);
            W = MaskW.*max(epsilon, W./ColumnMax );
            H = MaskH.*max(epsilon, H.*ColumnMax' );
        end
        
        WFinal(:,(1+CumSumIR{SourceCounter}(SourceInstance)):CumSumIR{SourceCounter}(SourceInstance+1)) = W;
        
        if SourceCounter == 1 %HERE
            HFinal{SourceInstance} = H;
        end
    end
end

%Weights initialization
InitialWeights = cell(1,C+1);
InitialWeights{1} = WFinal;

if strcmp(DiscriminativePropagation, 'NoContext')
    WDiscriminative = WFinal(end-m+1:end,:)./max(WFinal(end-m+1:end,:)); %No context, normalize
end

if strcmp(DiscriminativePropagation, 'Context')
    WDiscriminative = WFinal./max(WFinal); %Context, already normalized
end

for DiscriminativeLayerCounter = 2:C+1
    InitialWeights{DiscriminativeLayerCounter} = WDiscriminative;
end 
end
end

