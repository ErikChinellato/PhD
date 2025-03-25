function [States,MeasurementMinusCStates,GainMeasurementMinusCFs,MeasurementMinusCFs,FStateDynInputs] = PropagateInput(Inputs,Measurements,FirstState,Dynamic,F,NetWeights,NetParameters)
%PROPAGATEINPUT: Propagates the Inputs vector cell (u) and Measurements
%vector cell (y) throught the net. They are cells of size (1,NetParameters.Layers).
%F is the 'VARMION' function block. The output is the States vector cell (x),
%a cell of size (1,NetParameters.Layers+1). States{1} is given as in input. 
%Some additional output MeasurementMinusCFs, GainMeasurementMinusCFs, 
%MeasurementMinusCFs, FStatePInputs is saved for later efficiency during backpropagation.

%Variables
Layers = NetParameters.Layers;
C = NetParameters.C;
SharedWeights = NetParameters.SharedWeights;

%Setup output
States = cell(1,Layers+1);
MeasurementMinusCStates = cell(1,Layers);
GainMeasurementMinusCFs = cell(1,Layers);

%Cleaner propagation for test phase
if nargout > 3
    MeasurementMinusCFs = cell(1,Layers);
    FStateDynInputs = cell(Layers);
end

%Initialize the first state
States{1} = FirstState;

%Propagate
for Layer = 1:Layers
    if strcmp(SharedWeights,'Yes')
        Indx = 1;
    end

    if strcmp(SharedWeights,'No')
        Indx = Layer;
    end

    FStateDynInput = F(States{Layer},NetWeights{end}{Dynamic},Inputs{Layer},NetWeights{end}{end},Layer,NetParameters);
    MeasurementMinusCF = Measurements{Layer+1} - C*FStateDynInput;
    GainMeasurementMinusCF = NetWeights{Indx}*MeasurementMinusCF;
    
    %Save output
    States{Layer+1} = FStateDynInput + GainMeasurementMinusCF;
    MeasurementMinusCStates{Layer} = Measurements{Layer+1}-C*States{Layer+1};
    GainMeasurementMinusCFs{Layer} = GainMeasurementMinusCF;

    %Cleaner propagation for test phase
    if nargout > 3
        MeasurementMinusCFs{Layer} = MeasurementMinusCF;
        FStateDynInputs{Layer} = FStateDynInput;
    end
end

end

