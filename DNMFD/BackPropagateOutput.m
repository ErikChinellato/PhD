function Grads = BackPropagateOutput(X,HList,CleanSources,NetWeights,Grads,NetParameters)
%BACKPROPAGATEOUTPUT: Computes the gradients of the loss function with
%respects to the weights.

%Variables
S = NetParameters.Sources;
C = NetParameters.DiscriminativeLayers;

CleanSource = CleanSources{1};

%% LAST DISCRIMINATIVE LAYER
Weights = NetWeights(:,end);
H = HList(:,end);

%Compute gradients for H, W and add the latter to the current layer's minibatch sum
[GradH,Grads(:,:,end)] = ComputeLastGrad(X,CleanSource,Weights,H,Grads(:,:,end),NetParameters);

%% INTERMEDIATE DISCRIMINATIVE LAYERS
for i = 1:C-1
    Weights = NetWeights(:,end-i);
    H = HList(:,end-i);
    
    %Compute some reoccurring components
    [GradHCompA,GradHCompB,GradWCompA,GradWCompB,GradWCompC] = GradsComponents(X,Weights,H,GradH,NetParameters);
    %Update gradients for W with current (next outer layer) gradients of H
    %and add them to the current layer's minibatch sum
    Grads(:,:,end-i) = UpdateGradW(GradWCompA,GradWCompB,GradWCompC,Grads(:,:,end-i),NetParameters);
    
    %Update gradients for H with current (next outer layer) gradients of H UNLESS IN LAST ITERATION
    if i ~= C-1 %Not necessary, it wouldnt be used anyway, but less computations
        GradH = UpdateGradH(GradHCompA,GradHCompB,NetParameters);
    end
end

end


