function TensorizedGains = ConstructTensorizedGains(NetWeights,NetParameters)
%CONSTRUCTTENSORIZEDGAINS: Inserts the kalman gains into a 3-D tensor.

%Variables
Layers = NetParameters.Layers;
StateDimension = NetParameters.StateDimension;
ObservationDimension = NetParameters.ObservationDimension;

TensorizedGains = zeros(StateDimension,ObservationDimension,Layers);

%Assemble tensor
for Layer = 1:Layers
    TensorizedGains(:,:,Layer) = NetWeights{Layer};
end
end

