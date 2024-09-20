function Grads = InitializeGrads(NetParameters,NetWeights)
%INITIALIZEGRADS: Initializes the gradients to zero.

%Variables
C = NetParameters.DiscriminativeLayers;
S = NetParameters.Sources;

%Initialize gradients
Grads = cell(2,S,C);

for Source = 1:S
    for Layer = 1:C
        for Sign = 1:2
            Grads{Sign,Source,Layer} = zeros( size(NetWeights{Source,Layer}) );
        end
    end
end

end

