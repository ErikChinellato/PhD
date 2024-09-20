function [GradHCompA,GradHCompB,GradWCompA,GradWCompB,GradWCompC] = GradsComponents(X,Weights,H,GradH,NetParameters)
%GRADSCOMPONENTS: Computes the components of the gradients with respects to both the
%weights and H matrix ('s rows).

%Variables
S = NetParameters.Sources;
Epsilon = NetParameters.Epsilon;

%Set up output
GradHCompA = cell(2,S);
GradHCompB = cell(2,S);
GradWCompA = cell(2,S);
GradWCompB = cell(2,S);
GradWCompC = cell(2,S);

%Compute some reoccurring components for efficiency
XHat = ConstructXHat( Weights, H );
XdXHat = X./( XHat + Epsilon);
XdXHatSq = X./( XHat.^2 + Epsilon );

GradHdDen = cell(2,S);
HGradHdDen = cell(2,S);
HGradHdDenSq = cell(2,S);
SumConvXdXhat = cell(1,2);

for Source = 1:S
    Den = sum( Weights{Source}, 'all' );
    for Sign = 1:2
        GradHdDen{Sign,Source} = GradH{Sign,Source}./( Den + Epsilon);
        HGradHdDen{Sign,Source} = GradHdDen{Sign,Source}.*H{Source};
        HGradHdDenSq{Sign,Source} = HGradHdDen{Sign,Source}./( Den + Epsilon);
    end
end

for Sign = 1:2
    SumConvXdXhat{Sign} = ConstructXHat( Weights, HGradHdDen(Sign,:) ).*XdXHatSq;
end

%Compute the components of the gradients
for Source = 1:S
    for Sign = 1:2
        GradHCompA{Sign,Source} = GradHdDen{Sign,Source}.*Convolve( Weights{Source}, XdXHat );
        GradHCompB{Sign,Source} = Convolve( Weights{Source}, SumConvXdXhat{3-Sign} );
        GradWCompA{Sign,Source} = XdXHat*StackShiftMat( HGradHdDen{Sign,Source}, size(Weights{Source},2) );
        GradWCompB{Sign,Source} = sum( Convolve(Weights{Source},  HGradHdDenSq{3-Sign,Source} ), 'all' );
        GradWCompC{Sign,Source} = SumConvXdXhat{3-Sign}*StackShiftMat( H{Source}, size(Weights{Source},2) );
    end
end

end

