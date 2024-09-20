function NetWeights = UpdateWeights(NetWeights,GradListSum,NetParameters,Ind)
%UPDATEWEIGHTS: Updates the weights of the network.

%Variables
C = NetParameters.DiscriminativeLayers;
S = NetParameters.Sources;
ModifyLastNotSourcesOfInterestWeights = NetParameters.ModifyLastNotSourcesOfInterestWeights;
Epsilon = NetParameters.Epsilon;

%% LAST DISCRIMINATIVE LAYER
%Update W (final reconstruction layer)
if strcmp(ModifyLastNotSourcesOfInterestWeights,'Yes')
    NotSourcesOfInterest = NetParameters.NotSourcesOfInterest;
    for Source = NotSourcesOfInterest
        %Update W with usual Multiplicative Update rule for NMF
        NetWeights{Source,end} =  NetWeights{Source,end}.*( GradListSum{2,Source,end}./( GradListSum{1,Source,end} + Epsilon ) );
        NetWeights{Source,end} = max( Epsilon, NetWeights{Source,end}./max(NetWeights{Source,end}) ); %Rescale for stability and project
    end
end

if strcmp(ModifyLastNotSourcesOfInterestWeights,'No')
    %Do nothing
end

%% INTERMEDIATE DISCRIMINATIVE LAYERS
for i = 1:C-1
    for Source = 1:S
        %Update W with usual Multiplicative Update rule for NMF
        NetWeights{Source,end-i} = NetWeights{Source,end-i}.* ( GradListSum{2,Source,end-i}./( GradListSum{1,Source,end-i} + Epsilon ) );
        %NetWeights{Source,end-i} = max( Epsilon, NetWeights{Source,end-i} - Ind*(GradListSum{1,Source,end-i}-GradListSum{2,Source,end-i}) );
        NetWeights{Source,end-i} = max( Epsilon, NetWeights{Source,end-i}./max(NetWeights{Source,end-i}) ); %Rescale for stability and project
    end
end 

%% NON-DISCRIMINATIVE LAYERS
%Do nothing, the weights are fixed

end

