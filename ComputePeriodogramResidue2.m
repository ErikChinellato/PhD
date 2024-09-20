function PeriodogramResidues = ComputePeriodogramResidue2(MeasurementMinusCStates,MeasurementMinusCFs)
%COMPUTEPERIODOGRAMRESIDUE: Computes the periodogram residues for both
%correctors and predictors at every layers

Layers = length(MeasurementMinusCStates);

CorrectorResidues = [];
PredictorResidues = [];

for Layer = 1:Layers
    CorrectorResidues = [CorrectorResidues,MeasurementMinusCStates{Layer}'];
    PredictorResidues = [PredictorResidues,MeasurementMinusCFs{Layer}'];
end

CorrectorPeriodogram = TestBartlett(CorrectorResidues);
PredictorPeriodogram = TestBartlett(PredictorResidues);

CorrectorPeriodogramResidues = norm(CorrectorPeriodogram-linspace(0,1,length(CorrectorPeriodogram)));
PredictorPeriodogramResidues = norm(PredictorPeriodogram-linspace(0,1,length(PredictorPeriodogram)));

PeriodogramResidues = [CorrectorPeriodogramResidues,PredictorPeriodogramResidues];
end

