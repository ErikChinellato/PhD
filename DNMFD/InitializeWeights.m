function InitialWeights = InitializeWeights(CleanSources,NetParameters)
%INITIALIZEWEIGHTS: Initializes the net's weights.

%Variables
C = NetParameters.DiscriminativeLayers;
S = NetParameters.Sources;
SparsePen = NetParameters.SparsePenalty;
Ranks = NetParameters.Ranks;
Initialization = NetParameters.Initialization;
Beta = NetParameters.InitializationBeta;
MaxIt = NetParameters.InitializationMaxIt;
Epsilon = NetParameters.Epsilon;

%Weights initialization
InitialWeights = cell(S,C+1);

%Cycle over clean sources
for Source = 1:S
    X = CleanSources{Source};

    if strcmp(Initialization,'NMF')
        %Here we learn the basis functions with NMF
        H = max(  Epsilon, rand(Ranks(Source),size(X,2)) );
        W = max( Epsilon, X(:,1:Ranks(Source)) );
        for ItCounter = 1:MaxIt
            %Update H
            H = UpdateInitH(X,W,H,Beta,SparsePen,Epsilon);
            %Update W
            W = UpdateInitW(X,W,H,Beta,Epsilon);
            %Normalize W and H
            ColumnMax = max(W);
            W = max( Epsilon, W./ColumnMax );
            H = max( Epsilon, H.*ColumnMax' );
        end

        WInit = W;
    end

    if strcmp(Initialization,'AsIs')
        %Here we use the clean source itself
        WInit = max( Epsilon, X(:,1:Ranks(Source))./max(X(:,1:Ranks(Source))) );
    end

    %Set weights
    for Layer = 1:C+1
        InitialWeights{Source,Layer} = WInit;
    end
end
   
end


