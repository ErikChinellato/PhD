function GradListSum = BackPropagateOutput(X,HList,CleanSources,PropagationOptions,NetParameters,CurrentWeights,GradListSum)
%BACKPROPAGATEOUTPUT: backpropagation algorithm for the net. It computes the
%positive and negative gradients of the loss function with respects to the
%weights CurrentWeights using the propagated input HList. The gradients are
%then stored (added layerwise) in GradListSum for minibatch updating.
%CleanSources is a cell array of size (1,NetParameters.Sources), containing
%in each cell the (single, not concatenated) clean (as in, not mixed) source 
%that we want to reconstruct from X (AS OF RIGHT NOW ONLY THE FIRST ONE
%WILL BE RECONSTRUCTED, HARDCODED LIKE THIS).

%% VARIABLES INITIALIZATION
%Variables
C = NetParameters.DiscriminativeLayers;
SparsePen = NetParameters.SparsePenalty;
T = NetParameters.ContextFrames;
FactRanks = NetParameters.Ranks;
R = sum(FactRanks);
InitializationRanks = PropagationOptions.InitializationRanks;
Objective = NetParameters.Objective;
DiscriminativePropagation = NetParameters.DiscriminativePropagation;
ForwardAndBackPropagation = NetParameters.ForwardAndBackPropagation;
ForwardPropagationProjection = NetParameters.ForwardPropagationProjection;

%We only reconstruct one source (the first one)
Source = CleanSources{1};

%Propagation and backpropagation options
if ~isfield(PropagationOptions,'Epsilon')
    epsilon = 2^-52;
else
    epsilon = PropagationOptions.Epsilon;
end

if strcmp(DiscriminativePropagation, 'NoContext')
    %The weights in the discriminative layers do not have context frames
    XDiscriminative = X;
    SourceDiscriminative = Source;
    
    [RowNum, ColNum] = size(XDiscriminative);
    MaskH = ones(R,ColNum);
    ProjMat = eye(R);
end

if strcmp(DiscriminativePropagation, 'Context')
    %The weights in the discriminative layers have context frames
    [m,N] = size(X);
    XDiscriminative = ConstructContextMat(X,m,N,T);
    SourceDiscriminative = ConstructContextMat(Source,m,N,T);
    
    [RowNum, ColNum] = size(XDiscriminative);
    
    if strcmp(ForwardAndBackPropagation, 'NoCausalityPreserving')
        %Here we do not preserve the time relations present in the
        %dictionary atoms: all atoms can be activated at any time
        MaskH = ones(R,ColNum);
        ProjMat = eye(R);
    end
    
    if strcmp(ForwardAndBackPropagation, 'CausalityPreserving')
        %Here we preserve the time relations present in the dictionary atoms:
        %at a time t we cannot activate atoms relative to times < t
        MaskH = ones(R,ColNum);
        
        % InitRankOffset = 0;
        % for SourceInd = 1:length(FactRanks)
        %     CSInitRanks = [InitRankOffset, InitRankOffset + cumsum(InitializationRanks{SourceInd})];
        %     for RankInd = 1:length(InitializationRanks{SourceInd})
        %         for RowInd = 1:InitializationRanks{SourceInd}(RankInd)-1
        %             MaskH(CSInitRanks(RankInd)+RowInd,RowInd+1:ColNum) = 0; 
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

%% LAST DISCRIMINATIVE LAYER
W = CurrentWeights{end};
H = HList{end};

%Compute last gradients for H
[PosGradH,NegGradH,PosGradW,NegGradW] = ComputeLastGrad(XDiscriminative,W,H,SourceDiscriminative,epsilon,InitializationRanks,FactRanks,Objective);

%Add the gradients to the list for minibatch updating
GradListSum{1,end} = GradListSum{1,end} + NegGradW;
GradListSum{2,end} = GradListSum{2,end} + PosGradW;

%% INTERMEDIATE DISCRIMINATIVE LAYERS
for i = 1:C-1
    W = CurrentWeights{end-i};
    H = HList{end-i};
    
    %Take pages 26 and 27 of the paper 'Deep Unfolding: Model-Based
    %Inspiration of Novel Deep Architectures' as reference for the formulae
    %to follow. Match each addend to its code name.
    %
    % [grad H_k E]+ = PosGradH = PosGradHCompA + W_k'*PosGradHCompB (depends on PosGradH, NegGradH at the previous iterate)
    % [grad H_k E]- = NegGradH = NegGradHCompA + W_k'*NegGradHCompB (depends on PosGradH, NegGradH at the previous iterate)
    %
    % [grad W_k E]+ = PosgradW = PosGradWCompA + PosGradHCompB*H_k' + GradWCompC + PosGradWCompD (depends on PosGradH, NegGradH at the previous iterate)
    % [grad W_k E]- = NegGradW = NegGradWCompA + NegGradHCompB*H_k' + GradWCompC + NegGradWCompD (depends on PosGradH, NegGradH at the previous iterate)
    %
    % Lastly, check in ReoccurringComponents how each component is
    % constructed and verify it matches the paper. 
    % BEWARE: there are some mistakes in the formulae of the paper, mainly in the terms
    % W'*1_{NxR}+mu at the denominator. They should be W'*1_{NxT}+mu and should appear
    % at the denominator of an H matrix, not a W matrix. Also, the matrix
    % 1_{RxT} at the beginning of ComponentD for PosGradW and NegGradW
    % should be 1_{NxT}. 
    
    %Compute some reoccurring components
    [PosGradHCompA,PosGradHCompB,NegGradHCompA,NegGradHCompB,PosGradWCompA,NegGradWCompA,GradWCompC,PosGradWCompD,NegGradWCompD] = ReoccurringComponents(XDiscriminative,W,H,ProjMat*(MaskH.*PosGradH),ProjMat*(MaskH.*NegGradH),epsilon,SparsePen,RowNum,ColNum);
    %Update gradients for W with current (next outer layer) gradients of H
    [PosGradW,NegGradW] = UpdateGradW(PosGradHCompB,NegGradHCompB,PosGradWCompA,NegGradWCompA,GradWCompC,PosGradWCompD,NegGradWCompD,H);
    
    %Add the gradients to the list for minibatch updating
    GradListSum{1,end-i} = GradListSum{1,end-i} + NegGradW;
    GradListSum{2,end-i} = GradListSum{2,end-i} + PosGradW;
    
    %Update gradients for H with current (next outer layer) gradients of H UNLESS IN LAST ITERATION
    if i ~= C-1 %Not necessary, it wouldnt be used anyway, but less computations
        [PosGradH,NegGradH] = UpdateGradH(PosGradHCompA,PosGradHCompB,NegGradHCompA,NegGradHCompB,W);
    end
end

end

