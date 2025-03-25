function SparseDynMat = InitializeSparseDynamicsMat(DictionaryDimension,StateDimension,Model,Experiment)
%INITIALIZASPARSEDYNAMICSMAT: Initializes the sparse dynamics matrix.

if strcmp(Experiment,'1')
    SparseDynMat = sparse(zeros(DictionaryDimension,StateDimension));
end

if strcmp(Experiment,'2')
    SparseDynMat = sparse(zeros(DictionaryDimension,StateDimension));
end

if strcmp(Experiment,'3')
    SparseDynMat = sparse(zeros(DictionaryDimension,StateDimension));
end

if strcmp(Experiment,'4')
    SparseDynMat = sparse(zeros(DictionaryDimension,StateDimension));
end

if strcmp(Experiment,'5')
    SparseDynMat = sparse(zeros(DictionaryDimension,StateDimension));
end

if strcmp(Experiment,'6')
    SparseDynMat = sparse(zeros(DictionaryDimension,StateDimension));
end
end

