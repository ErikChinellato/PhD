function NewWeights = UpdateWeights(PropagationOptions,NetParameters,CurrentWeights,GradListSum)
%UPDATEWEIGHTS: updates the weights in CurrentWeigths using the gradients
%stored in GradListSum. The updated weights are NewWeights.

%% VARIABLES INITIALIZATION
%Variables
C = NetParameters.DiscriminativeLayers;
T = NetParameters.ContextFrames;
FactRanks = NetParameters.Ranks;
InitializationRanks = PropagationOptions.InitializationRanks;
DiscriminativePropagation = NetParameters.DiscriminativePropagation;
ForwardandBackPropagation = NetParameters.ForwardAndBackPropagation;
MaskWSize = size(GradListSum{1,end});

%Propagation and backpropagation options
if ~isfield(PropagationOptions,'Epsilon')
    epsilon = 2^-52;
else
    epsilon = PropagationOptions.Epsilon;
end

if strcmp(DiscriminativePropagation, 'NoContext')
    %The weights in the discriminative layers do not have context frames
    MaskW = ones(MaskWSize);
end
if strcmp(DiscriminativePropagation, 'Context')
    %The weights in the discriminative layers have context frames
    if strcmp(ForwardandBackPropagation, 'NoCausalityPreserving')
        %Here we do not preserve the hankel-like structure of the discriminative
        %weights: the complete dictionary can be modified
        MaskW = ones(MaskWSize);
    end
    
    if strcmp(ForwardandBackPropagation, 'CausalityPreserving')
        %Here we preserve the hankel-like structure of the discriminative
        %weights by projecting: only certain blocks of the dictionary can be modified (non-zero)
        MaskW = ones(MaskWSize);
        m = floor(MaskWSize(1)/T);
        
        InitRankOffset = 0;
        for SourceInd = 1:length(FactRanks)
            CSInitRanks = [InitRankOffset, InitRankOffset + cumsum(InitializationRanks{SourceInd})];
            for BlockInd = 1:T-1
                for RankInd = 1:length(InitializationRanks{SourceInd})      
                        MaskW(m*(BlockInd-1)+1:m*BlockInd,CSInitRanks(RankInd)+1:CSInitRanks(RankInd)+T-BlockInd) = 0; 
                end
            end
            InitRankOffset = CSInitRanks(end);
        end
    end   
end

NewWeights = cell(1,C+1); %size(CurrentWeights) = [1,C+1]

%% LAST DISCRIMINATIVE LAYER
%W = CurrentWeights{end};
%Update W (final reconstruction layer, it remains fixed)
NewWeights{end} = CurrentWeights{end};
%W = CurrentWeights{end};
%NewWeights{end} =  W.*(GradListSum{1,end}./(GradListSum{2,end}+epsilon));
%NewWeights{end} = MaskW.*max( epsilon, NewWeights{end}./max(NewWeights{end}) ); %Rescale for stability and project

%% INTERMEDIATE DISCRIMINATIVE LAYERS
for i = 1:C-1
    W = CurrentWeights{end-i};
    %Update W with usual Multiplicative Update rule for NMF
    NewWeights{end-i} = W.*(GradListSum{1,end-i}./(GradListSum{2,end-i}+epsilon));
    NewWeights{end-i} = MaskW.*max( epsilon, NewWeights{end-i}./max(NewWeights{end-i}) ); %Rescale for stability and project
end 

%% NON-DISCRIMINATIVE LAYERS
NewWeights{1} = CurrentWeights{1};

end

