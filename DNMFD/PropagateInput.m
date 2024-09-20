function HList = PropagateInput(X,NetParameters,NetWeights)
%PROPAGATEINPUT: Computes the forward run of the net.

%Variables
K = NetParameters.Layers;
C = NetParameters.DiscriminativeLayers;
S = NetParameters.Sources;
SparsePen = NetParameters.SparsePenalty;
Epsilon = NetParameters.Epsilon;

%Set up initial H matrices
HList = cell(S,K);
for Source = 1:S
    HList{Source,1} = max( Epsilon, 1e-1*ones(1,size(X,2)) );%max( Epsilon, rand(1,size(X,2)) );
end

%% FIRST K-C (NON-DISCRIMINATIVE) PROPAGATION LAYERS (1 to K-C)
%Compute recurrent components for efficiency
Dens = cell(1,S);
for Source = 1:S
    Dens{Source} = sum( NetWeights{Source,1},'all' ) + SparsePen; 
end

%Propagation
for HCounter = 1:K-C
    XHat = ConstructXHat(NetWeights(:,1),HList(:,HCounter));
    XdXHat = X./( XHat + Epsilon );
    for Source = 1:S
        HList{Source,HCounter+1} = UpdatePropNonDiscrH(XdXHat,NetWeights{Source,1},HList{Source,HCounter},Dens{Source},Epsilon);
    end
end

%% LAST C-1 (DISCRIMINATIVE) PROPAGATION LAYERS (K-C+1 to K-1)
%Propagation
WCounter = 2;
for HCounter = K-C+1:K-1
    XHat = ConstructXHat(NetWeights(:,WCounter),HList(:,HCounter));
    XdXHat = X./( XHat + Epsilon );
    for Source = 1:S
        HList{Source,HCounter+1} = UpdatePropDiscrH(XdXHat,NetWeights{Source,WCounter},HList{Source,HCounter},SparsePen,Epsilon);
    end
    WCounter = WCounter + 1;
end

end

