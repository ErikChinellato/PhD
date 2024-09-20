function Grads = BackpropagateOutput(YClean,XClean,SparseReps,Patches,PatchesDenoised,Lambdas,Mus,Nus,Csis,dSTs,DSparseRepmPatches,XDen,Grads,NetWeights,NetParameters)
%BACKPROPAGATEOUTPUT: Backpropagates the output and computes the gradients
%for the net's parameters. Grads is a dictionary with keys 'Dict', 'C',
%'W', 'b0A0', 'b1A1', 'b2A2', 'b3A3' associated to the corresponding gradients.
%YClean is the desired clean image output and XClean is the net's output.

%Variables
T = NetParameters.Layers;
PatchSize = NetParameters.PatchSize;
SubImagesSize = NetParameters.SubImagesSize;
DictionarySize = NetParameters.DictionarySize;
SharedWeights = NetParameters.SharedWeights;
MLPLastActivation = NetParameters.MLPLastActivation;

M = DictionarySize^2;
PatchNum = (SubImagesSize-PatchSize+1)^2;

%Compute last gradients
[Grads,GradSparseReps] = ComputeLastGrad(YClean,XClean,Grads,SparseReps(end,:),NetWeights{'W'}{1},NetWeights{'Dict'}{end},PatchesDenoised,XDen,PatchSize,SubImagesSize,PatchNum);

%Update the recursive gradients with respects to sparse representations and
%compute the other gradients along the way
for t = T:-1:1 %Morally this runs from T-1 to 0
    if strcmp(SharedWeights,'Yes')
        Indx = 1;
    end
    if strcmp(SharedWeights,'No')
        Indx = t;
    end
    
    %Compute (patch independent) reoccurring components for efficiency
    OnedC = 1/NetWeights{'C'}{Indx};
    OnedCSq = OnedC/NetWeights{'C'}{Indx};
    UpdateMat = eye(M) - OnedC*NetWeights{'Dict'}{Indx}'*NetWeights{'Dict'}{Indx};

    A1 = NetWeights{'b1A1'}{Indx}(:,2:end);
    A2 = NetWeights{'b2A2'}{Indx}(:,2:end);
    A3 = NetWeights{'b3A3'}{Indx}(:,2:end);

    for PatchInd = 1:PatchNum
        %Compute reoccurring components for efficiency
        GSRdSTTheta = GradSparseReps{PatchInd}.*dSTs{1,t,PatchInd};
        GSRdSTVec = GradSparseReps{PatchInd}.*dSTs{2,t,PatchInd};
        SumGSRdSTTheta = OnedC*sum(GSRdSTTheta);

        if strcmp(MLPLastActivation,'Identity')
            ColCsi = 1;
        end
        if strcmp(MLPLastActivation,'ReLU')
            ColCsi = dReLU( NetWeights{'b3A3'}{Indx}*[1;Csis{Indx,PatchInd}] );
        end
        if strcmp(MLPLastActivation,'Sigmoid')
            ColCsi = dSigmoid( NetWeights{'b3A3'}{Indx}*[1;Csis{Indx,PatchInd}] );
        end
        
        ColMu = dReLU( NetWeights{'b2A2'}{Indx}*[1;Mus{Indx,PatchInd}] ).*( A3'*ColCsi) ;
        ColNu = dReLU( NetWeights{'b1A1'}{Indx}*[1;Nus{Indx,PatchInd}] ).*( A2'*ColMu) ;
        ColY = dReLU( NetWeights{'b0A0'}{Indx}*[1;Patches{PatchInd}] ).*( A1'*ColNu );
        
        %Update gradients of the net with current patch contribution
        Grads{'Dict'}{Indx} = Grads{'Dict'}{Indx} + (-OnedC)*( DSparseRepmPatches{t,PatchInd}*GSRdSTVec' + ( NetWeights{'Dict'}{Indx}*GSRdSTVec )*SparseReps{t,PatchInd}' );
        Grads{'C'}{Indx} = Grads{'C'}{Indx} + OnedCSq*sum( GSRdSTVec.*( NetWeights{'Dict'}{Indx}'*DSparseRepmPatches{t,PatchInd} ) - Lambdas{Indx,PatchInd}*GSRdSTTheta );

        Grads{'b3A3'}{Indx} = Grads{'b3A3'}{Indx} + SumGSRdSTTheta*ColCsi*[1,Csis{Indx,PatchInd}'];
        Grads{'b2A2'}{Indx} = Grads{'b2A2'}{Indx} + SumGSRdSTTheta*ColMu*[1,Mus{Indx,PatchInd}'];
        Grads{'b1A1'}{Indx} = Grads{'b1A1'}{Indx} + SumGSRdSTTheta*ColNu*[1,Nus{Indx,PatchInd}'];
        Grads{'b0A0'}{Indx} = Grads{'b0A0'}{Indx} + SumGSRdSTTheta*ColY*[1,Patches{PatchInd}']; 
    
        %Update gradient of sparse representation associated to current patch
        GradSparseReps{PatchInd} = UpdateMat*GSRdSTVec;
    end
end

end

