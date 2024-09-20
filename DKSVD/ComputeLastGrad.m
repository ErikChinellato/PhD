function [Grads,GradSparseReps] = ComputeLastGrad(YClean,XClean,Grads,LastSparseReps,W,Dict,PatchesDenoised,XDen,PatchSize,SubImagesSize,PatchNum)
%COMPUTELASTGRAD: Updates Grads by inserting the gradients at the last
%layer and computes GradSparseReps, the gradients with respects to the
%sparse representations at the last layer.

%Variables and reoccurring components
GradSparseReps = cell(1,PatchNum);

Scale = 2/( SubImagesSize^2 );
XmYdXDen = (XClean-YClean)./XDen;
XXmYdXDen = XmYdXDen.*XClean;

%Cycle over patches and update gradients at last layer
for PatchInd = 1:PatchNum
    RXmYdXDen = R(XmYdXDen,PatchSize,PatchInd);
    RXXmYdXDen = R(XXmYdXDen,PatchSize,PatchInd);
    WRXmYdXDen = W.*RXmYdXDen;

    GradSparseReps{PatchInd} = Scale*Dict'*WRXmYdXDen;
    
    Grads{'W'}{1} = Grads{'W'}{1} + Scale*( PatchesDenoised{PatchInd}.*RXmYdXDen - RXXmYdXDen );
    Grads{'Dict'}{end} = Grads{'Dict'}{end} + Scale*WRXmYdXDen*LastSparseReps{PatchInd}';
end

end

