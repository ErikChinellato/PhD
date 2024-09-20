import numpy as np
from scipy.sparse import csr_matrix

def InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment):
    """
    Initializes the sparse dynamics matrix.
    """
    if Experiment == '4':
        SparseDynMat = np.zeros((DictionaryDimension, StateDimension)).astype('float64')
    return SparseDynMat