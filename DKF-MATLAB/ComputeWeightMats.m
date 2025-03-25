function [MeasurementWeightMats,PredictorWeightMats,MeasurementWeightMatsSym,PredictorWeightMatsSym] = ComputeWeightMats(NetParameters)
%COMPUTEWEIGHTMATS: Compute the weight matrices used for residue scaling

%Variables
Layers = NetParameters.Layers;
StateDimension = NetParameters.StateDimension;
ObservationDimension = NetParameters.ObservationDimension;
WeightMats = NetParameters.WeightMats;

MeasurementWeightMats = cell(1,Layers);
PredictorWeightMats = cell(1,Layers);

%Cleaner propagation for test phase
if nargout > 2
    MeasurementWeightMatsSym = cell(1,Layers);
    PredictorWeightMatsSym = cell(1,Layers);
end

%Compute weight matrices
if strcmp(WeightMats,'Identity')
    for Layer = 1:Layers
        MeasurementWeightMats{Layer} = eye(ObservationDimension);
        PredictorWeightMats{Layer} = eye(StateDimension);

        %Cleaner propagation for test phase
        if nargout > 2
            MeasurementWeightMatsSym{Layer} = eye(ObservationDimension);
            PredictorWeightMatsSym{Layer} = eye(StateDimension);
        end
    end
end

if strcmp(WeightMats,'Input')
    Experiment = NetParameters.Experiment;

    try
        MeasurementWeightMats = importdata(['MeasurementWeightMatsExp',Experiment,'.mat']);
    catch ME
        if (strcmp(ME.identifier,'MATLAB:importdata:FileNotFound'))
            msg = ['File "MeasurementWeightMats',Experiment,'.mat" not found in current path.'];
            causeException = MException('MATLAB:myCode:FileNotFound',msg);
             ME = addCause(ME,causeException);
        end
        rethrow(ME)
    end 

    try
        PredictorWeightMats = importdata(['PredictorWeightMatsExp',Experiment,'.mat']);
    catch ME
        if (strcmp(ME.identifier,'MATLAB:importdata:FileNotFound'))
            msg = ['File "MeasurementWeightMats',Experiment,'.mat" not found in current path.'];
            causeException = MException('MATLAB:myCode:FileNotFound',msg);
             ME = addCause(ME,causeException);
        end
        rethrow(ME)
    end 

    %Cleaner propagation for test phase
    if nargout > 2
        for Layer = 1:Layers
            MeasurementWeightMatsSym{Layer} = (1/2)*( MeasurementWeightMats{Layer} + MeasurementWeightMats{Layer}' );
            PredictorWeightMatsSym{Layer} = (1/2)*( PredictorWeightMats{Layer} + PredictorWeightMats{Layer}' );
        end
    end
end


end
