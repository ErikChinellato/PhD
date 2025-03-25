function [Grads,Moment1,Moment2] = InitializeGradsAndMoments(NetWeights,NetParameters)
%INITIALIZEGRADSANDMOMENTS: Initializes the gradients for the net's parameters to zero.

%Variables
Layers = NetParameters.Layers;
SharedWeights = NetParameters.SharedWeights;
HiddenDynamicsNumber = NetParameters.HiddenDynamicsNumber;

%Setup gradients
if strcmp(SharedWeights,'Yes')
    Grads = cell(1,2);

    Grads{1} = zeros(size(NetWeights{1}));
    for Dyn = 1:HiddenDynamicsNumber
        Grads{2}{Dyn} = zeros(size(NetWeights{2}{Dyn}));
    end
end

if strcmp(SharedWeights,'No')
    Grads = cell(1,Layers+1);

    for Layer = 1:Layers
        Grads{Layer} = zeros(size(NetWeights{Layer}));
    end
    for Dyn = 1:HiddenDynamicsNumber
        Grads{Layers+1}{Dyn} = zeros(size(NetWeights{Layers+1}{Dyn}));
    end
end

if nargout > 1
    Moment1 = Grads;
    Moment2 = Grads;
end
end

