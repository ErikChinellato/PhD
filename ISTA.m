function SparseMat = ISTA(DtDdC,DtTargetdC,SparseMat,Thresh,MaxIter)
%ISTA: Iterative Soft Thresholding Algorithm for sparse coding.

for Iter = 1:MaxIter
    SparseMat = SoftThresh( Thresh, SparseMat - DtDdC*SparseMat + DtTargetdC );
end

end

