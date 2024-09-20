function HList = PropagateInput(X,PropagationOptions,NetParameters,CurrentWeights)
%PROPAGATEINPUT: propagates the input (X,H_0) through the net and outputs
%the sequence Hlist = {H_k}_{k=1,...,K}.

%% VARIABLES INITIALIZATION
%Variables
K = NetParameters.Layers;
C = NetParameters.DiscriminativeLayers;
SparsePen = NetParameters.SparsePenalty;
FactRanks = NetParameters.Ranks;
R = sum(FactRanks);
InitializationRanks = PropagationOptions.InitializationRanks;
DiscriminativePropagation = NetParameters.DiscriminativePropagation;
ForwardAndBackPropagation = NetParameters.ForwardAndBackPropagation;
ForwardPropagationProjection = NetParameters.ForwardPropagationProjection;

HList = cell(1,K);

[m,N] = size(X);

%Propagation options for input
if ~isfield(PropagationOptions,'Epsilon')
    epsilon = 2^-52;
else
    epsilon = PropagationOptions.Epsilon;
end
if ~isfield(PropagationOptions,'HInput')
   HList{1} = max(epsilon,rand(R,N));
else
   HList{1} = PropagationOptions.HInput; 
end

%% FIRST K-C (NON-DISCRIMINATIVE) PROPAGATION LAYERS (1 to K-C)
T = NetParameters.ContextFrames;
ContextX = ConstructContextMat(X,m,N,T); 

%Compute fixed components for efficiency
ContextW = CurrentWeights{1};
Den = repmat(sum(ContextW)',1,N) + SparsePen; 

%Propagation
for HCounter = 1:K-C
    HList{HCounter+1} = UpdatePropContextH(ContextX,ContextW,HList{HCounter},epsilon,Den);
end

%% LAST C-1 (DISCRIMINATIVE) PROPAGATION LAYERS (K-C+1 to K-1)
if strcmp(DiscriminativePropagation, 'NoContext')
    %The mixture does not have context frames in the discriminative layers
    XDiscriminative = X; 
    MaskH = ones(R,N);
    ProjMat = eye(R);
end

if strcmp(DiscriminativePropagation, 'Context')
    %The mixture has context frames in the discriminative layers
    XDiscriminative = ContextX;
    
    if strcmp(ForwardAndBackPropagation, 'NoCausalityPreserving')
        %Here we do not preserve the time relations present in the
        %dictionary atoms: all atoms can be activated at any time
        MaskH = ones(R,N);
        ProjMat = eye(R);
    end
    
    if strcmp(ForwardAndBackPropagation, 'CausalityPreserving')
        %Here we preserve the time relations present in the dictionary atoms:
        %at a time t we cannot activate atoms relative to times different from t
        MaskH = ones(R,N);
        
        % InitRankOffset = 0;
        % for SourceInd = 1:length(FactRanks)
        %     CSInitRanks = [InitRankOffset, InitRankOffset + cumsum(InitializationRanks{SourceInd})];
        %     for RankInd = 1:length(InitializationRanks{SourceInd})
        %         for RowInd = 1:InitializationRanks{SourceInd}(RankInd)-1
        %             %MaskH(CSInitRanks(RankInd)+RowInd,RowInd+1:N) = 0; 
        %             MaskH(CSInitRanks(RankInd)+RowInd,1:RowInd-1) = 0;
        %         end
        %         MaskH(CSInitRanks(RankInd)+RowInd+1,1:RowInd) = 0;
        %     end
        %     InitRankOffset = CSInitRanks(end);
        % end
        
        if strcmp(ForwardPropagationProjection, 'NoEnergyPreserving')
            %Here we do not preserve the energy of mis-activated atoms: all atoms
            %active at different times with respect to their relative one, 
            %are set to zero
            ProjMat = eye(R);
        end
        
        if strcmp(ForwardPropagationProjection, 'EnergyPreserving')
            %Here we preserve the energy of mis-activated atoms: all atoms
            %active at different times with respect to their relative one, 
            %share the activation energy with the correct atom at that time
            %instant
            ProjMat = [];
            for SourceInd = 1:length(FactRanks)
               for RankInd = 1:length(InitializationRanks{SourceInd})
                   ProjMat = blkdiag(ProjMat,ones(InitializationRanks{SourceInd}(RankInd)));
               end
            end
            
        end
    end
end

%Propagation
WCounter = 2;
for HCounter = K-C+1:K-1
    HList{HCounter+1} = MaskH.*( ProjMat*UpdatePropH(XDiscriminative,CurrentWeights{WCounter},HList{HCounter},epsilon,SparsePen,N) );
    WCounter = WCounter + 1;
end
end

