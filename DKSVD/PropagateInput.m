function [XClean,SparseReps,Patches,PatchesDenoised,Lambdas,Mus,Nus,Csis,dSTs,DSparseRepmPatches,XDen] = PropagateInput(YNoisy,NetWeights,NetParameters)
%PROPAGATEINPUT: Propagates the input noisy (square) image YNoisy through the network. 
%Returns the denoised image XClean, the sparse representations SparseRep for each 
%patch in Patches at all times, the corresponding values for Lambda, Mu, Nu, and some 
%auxiliary arguments PatchesDenoised, dSTs, XDen.

%Variables
T = NetParameters.Layers;
PatchSize = NetParameters.PatchSize;
DictionarySize = NetParameters.DictionarySize;
SharedWeights = NetParameters.SharedWeights;
ClipPatchDenoised = NetParameters.ClipPatchDenoised;
Epsilon = NetParameters.Epsilon;

M = DictionarySize^2;

%This supports rectangular testing images
ImageSize = size(YNoisy);
PatchNum = prod( (ImageSize-PatchSize+1) );

%Setup output quantities
XNum = zeros(ImageSize);
XDen = zeros(ImageSize);
SparseReps = cell(T+1,PatchNum);
Patches = cell(1,PatchNum);
PatchesDenoised = cell(1,PatchNum);

%Cleaner propagation for testing phase
if nargout > 4
    dSTs = cell(2,T,PatchNum);
    DSparseRepmPatches = cell(T,PatchNum);
    if strcmp(SharedWeights,'Yes')
        Lambdas = cell(1,PatchNum);
        Mus = cell(1,PatchNum);
        Nus = cell(1,PatchNum);
        Csis = cell(1,PatchNum);
    end
    if strcmp(SharedWeights,'No')
        Lambdas = cell(T,PatchNum);
        Mus = cell(T,PatchNum);
        Nus = cell(T,PatchNum);
        Csis = cell(T,PatchNum);
    end
end

%Extract patch
for PatchInd = 1:PatchNum
    Patch = R(YNoisy,PatchSize,PatchInd);
    Patches{PatchInd} = Patch;

    %Propagate patch
    SparseRep = zeros(M,1);
    SparseReps{1,PatchInd} = SparseRep;
    for t = 1:T
        if strcmp(SharedWeights,'Yes')
            Indx = 1;
        end

        if strcmp(SharedWeights,'No')
            Indx = t;
        end
        
        %Setup quantities to save for efficiency
        [Lambda,Mu,Nu,Csi] = MLP(Patch,NetWeights{'b0A0'}{Indx},NetWeights{'b1A1'}{Indx},NetWeights{'b2A2'}{Indx},NetWeights{'b3A3'}{Indx},NetParameters);
        Lambdas{Indx,PatchInd} = Lambda;
        
        %Cleaner propagation for testing phase
        if nargout > 4
            Mus{Indx,PatchInd} = Mu;
            Nus{Indx,PatchInd} = Nu;
            Csis{Indx,PatchInd} = Csi;
        end
        
        DSparseRepmPatch = NetWeights{'Dict'}{Indx}*SparseRep - Patch;
        Theta = Lambda/NetWeights{'C'}{Indx};
        Vec = SparseRep - ( 1/NetWeights{'C'}{Indx} )*NetWeights{'Dict'}{Indx}'*DSparseRepmPatch;
        
        %Cleaner propagation for testing phase
        if nargout > 4 
            DSparseRepmPatches{t,PatchInd} = DSparseRepmPatch;
            [dSTs{1,t,PatchInd}, dSTs{2,t,PatchInd}] = dSoftThresh(Theta,Vec);
        end

        %Update sparse representation of patch
        SparseRep = SoftThresh(Theta,Vec);
        SparseReps{t+1,PatchInd} = SparseRep;
    end
    
    %Insert denoised patch
    PatchDenoised = NetWeights{'Dict'}{end}*SparseRep;
    if strcmp(ClipPatchDenoised,'Yes')
        PatchDenoised( PatchDenoised > 1 ) = 1;
        PatchDenoised( PatchDenoised < -1 ) = -1;
    end
    
    PatchesDenoised{PatchInd} = PatchDenoised;

    XNum = XNum + Rt(NetWeights{'W'}{1}.*PatchDenoised,ImageSize,PatchInd);
    XDen = XDen + Rt(NetWeights{'W'}{1},ImageSize,PatchInd);
end

%For stability
XDen( (XDen < Epsilon)&( XDen > -Epsilon ) ) = Epsilon;

%Assemble denoised image
XClean = XNum./XDen;
end

