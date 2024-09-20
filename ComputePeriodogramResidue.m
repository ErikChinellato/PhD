function PeriodogramResidues = ComputePeriodogramResidue(MeasurementMinusCStates,MeasurementMinusCFs)
%COMPUTEPERIODOGRAMRESIDUE: Computes the periodogram residues for both
%correctors and predictors at every layers

Layers = length(MeasurementMinusCStates);
ObservationDimension = length(MeasurementMinusCStates{1});

CorrectorResidues = zeros(ObservationDimension,Layers);
PredictorResidues = zeros(ObservationDimension,Layers);

for Layer = 1:Layers
    CorrectorResidues(:,Layer) = MeasurementMinusCStates{Layer};
    PredictorResidues(:,Layer) = MeasurementMinusCFs{Layer};
end

CorrectorPeriodogramResidues = zeros(1,ObservationDimension);
PredictorPeriodogramResidues = zeros(1,ObservationDimension);

for ObservedState = 1:ObservationDimension
    CorrectorPeriodogram = TestBartlett(CorrectorResidues(ObservedState,:));
    PredictorPeriodogram = TestBartlett(PredictorResidues(ObservedState,:));

    CorrectorPeriodogramResidues(ObservedState) = norm(CorrectorPeriodogram-linspace(0,1,length(CorrectorPeriodogram)));
    PredictorPeriodogramResidues(ObservedState) = norm(PredictorPeriodogram-linspace(0,1,length(PredictorPeriodogram)));
end

PeriodogramResidues = [CorrectorPeriodogramResidues,PredictorPeriodogramResidues];
end

