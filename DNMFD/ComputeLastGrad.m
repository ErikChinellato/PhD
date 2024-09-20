function [GradH,Grads] = ComputeLastGrad(X,CleanSource,Weights,H,Grads,NetParameters)
%COMPUTELASTGRAD: Computes the gradients at the last layer with respects to
%the H matrix ('s rows). GradH is a cell array of size (2,NetParameters.Sources)
%where the first row constains the positive gradients and the second row
%the negative gradients for each source.

%Variables
S = NetParameters.Sources;
ReconstructionPen = NetParameters.ReconstructionPenalty;
SourcesOfInterest = NetParameters.SourcesOfInterest;
NotSourcesOfInterest = NetParameters.NotSourcesOfInterest;
ModifyLastNotSourcesOfInterestWeights = NetParameters.ModifyLastNotSourcesOfInterestWeights;
Epsilon = NetParameters.Epsilon;
Objective = NetParameters.Objective;

GradHComp = cell(2,S);
GradH = cell(2,S);

%Compute reoccurring components for efficiency
XHat = ConstructXHat( Weights, H );
XSHat = ConstructXHat( Weights(SourcesOfInterest), H(SourcesOfInterest) );
XSHatdXHat = XSHat./( XHat + Epsilon );
XdXHat = X./( XHat + Epsilon );

SXdXHat = CleanSource.*XdXHat;
SXXSHatdXHatSq = SXdXHat.*XSHatdXHat;
XSHatXSqdXHatSq = XSHat.*( XdXHat.^2 );
XSHatSqXSqdXHatTh = XSHatXSqdXHatSq.*XSHatdXHat;

%% WienerFilterReconstruction
if strcmp(Objective,'WienerFilterReconstruction')
    %Set up components of the gradients for H
    GradHCompInterest{1} = XSHatXSqdXHatSq + SXXSHatdXHatSq + ReconstructionPen*XSHat;
    GradHCompInterest{2} = SXdXHat + XSHatSqXSqdXHatTh + ReconstructionPen*CleanSource;
    GradHCompNotInterest{1} = SXXSHatdXHatSq;
    GradHCompNotInterest{2} = XSHatSqXSqdXHatTh;
    
    for Sign = 1:2
        GradHComp(Sign,SourcesOfInterest) = GradHCompInterest(Sign);
        GradHComp(Sign,NotSourcesOfInterest) = GradHCompNotInterest(Sign);
    end
    
    %Compute the gradients for H
    for Source = 1:S
        for Sign = 1:2
            GradH{Sign,Source} = Convolve( Weights{Source}, GradHComp{Sign,Source} );
        end
    end

    if strcmp(ModifyLastNotSourcesOfInterestWeights,'Yes')
        GradWComp = cell(2,S);
        %Set up components of the gradients for W
        GradWCompNotInterest{1} = SXXSHatdXHatSq;
        GradWCompNotInterest{2} = XSHatSqXSqdXHatTh;

        for Sign = 1:2
            GradWComp(Sign,NotSourcesOfInterest) = GradWCompNotInterest(Sign);
        end

        %Compute the gradients for W
        for Source = NotSourcesOfInterest
            StackMat = StackShiftMat( H{Source}, size(Weights{Source},2) );
            for Sign = 1:2
                Grads{Sign,Source} = Grads{Sign,Source} + GradWComp{Sign,Source}*StackMat;
            end
        end
    end

    if strcmp(ModifyLastNotSourcesOfInterestWeights,'No')
        %Do nothing
    end
end

end

