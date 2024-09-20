function [Grads,Moment1,Moment2] = InitializeGradsAndMoments(NetWeights,NetParameters)
%INITIALIZEGRADS: Initializes the gradients for the net's parameters to zero.

%Variables
T = NetParameters.Layers;
SharedWeights = NetParameters.SharedWeights;

%Setup gradients
if strcmp(SharedWeights,'Yes')
    %Here the weights are shared between layers
    GradDict = cell(1);
    GradC = cell(1);
    GradW = cell(1);
    Gradb0A0 = cell(1);
    Gradb1A1 = cell(1);
    Gradb2A2 = cell(1);
    Gradb3A3 = cell(1);

    GradW{1} = zeros(size(NetWeights{'W'}{1}));

    GradDict{1} = zeros(size(NetWeights{'Dict'}{1}));
    GradC{1} = zeros(size(NetWeights{'C'}{1}));
    Gradb0A0{1} = zeros(size(NetWeights{'b0A0'}{1}));
    Gradb1A1{1} = zeros(size(NetWeights{'b1A1'}{1}));
    Gradb2A2{1} = zeros(size(NetWeights{'b2A2'}{1}));
    Gradb3A3{1} = zeros(size(NetWeights{'b3A3'}{1}));
end

if strcmp(SharedWeights,'No')
    %Here the weights are NOT shared between layers
    GradDict = cell(1,T+1);
    GradC = cell(1,T);
    GradW = cell(1);
    Gradb0A0 = cell(1,T);
    Gradb1A1 = cell(1,T);
    Gradb2A2 = cell(1,T);
    Gradb3A3 = cell(1,T);

    GradW{1} = zeros(size(NetWeights{'W'}{1}));

    for t = 1:T+1
        GradDict{t} = zeros(size(NetWeights{'Dict'}{t}));
        if t < T+1
            GradC{t} = zeros(size(NetWeights{'C'}{t}));
            Gradb0A0{t} = zeros(size(NetWeights{'b0A0'}{t}));
            Gradb1A1{t} = zeros(size(NetWeights{'b1A1'}{t}));
            Gradb2A2{t} = zeros(size(NetWeights{'b2A2'}{t}));
            Gradb3A3{t} = zeros(size(NetWeights{'b3A3'}{t}));
        end
    end

end

Grads = dictionary('Dict',{GradDict}, 'C',{GradC}, 'W',{GradW}, 'b0A0',{Gradb0A0}, 'b1A1',{Gradb1A1}, 'b2A2',{Gradb2A2}, 'b3A3',{Gradb3A3});

if nargout > 1
    Moment1 = Grads;
    Moment2 = Grads;
end

end

