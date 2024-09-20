function [NetWeights,Moment1,Moment2] = UpdateWeights(NetWeights,Grads,Moment1,Moment2,Iterate,NetParameters)
%UPDATEWEIGHTS: Updates the net's weights.

%Variables
T = NetParameters.Layers;
SharedWeights = NetParameters.SharedWeights;
LearningRate = NetParameters.LearningRate;
Optimizer = NetParameters.Optimizer;
NormalizeDictionary = NetParameters.NormalizeDictionary;
ProjectLastMLPWeights = NetParameters.ProjectLastMLPWeights;
Epsilon = NetParameters.Epsilon;

Keys = NetWeights.keys;

%Optimizer choice
if strcmp(Optimizer,'SGD')
    %Do nothing, use Grads
end

if strcmp(Optimizer,'Adam')
    %Modify Grads
    Beta1 = NetParameters.BetaMoment1;
    Beta2 = NetParameters.BetaMoment2;
    
    if strcmp(SharedWeights,'Yes')
        for KeyInd = 1:length(Keys)
            %Modify
            Moment1{Keys(KeyInd)}{1} = Beta1*Moment1{Keys(KeyInd)}{1} + ( 1 - Beta1 )*Grads{Keys(KeyInd)}{1};
            Moment2{Keys(KeyInd)}{1} = Beta2*Moment2{Keys(KeyInd)}{1} + ( 1 - Beta2 )*( Grads{Keys(KeyInd)}{1}.^2 );

            Moment1Hat = Moment1{Keys(KeyInd)}{1}/( 1 - Beta1^Iterate );
            Moment2Hat = Moment2{Keys(KeyInd)}{1}/( 1 - Beta2^Iterate );

            Grads{Keys(KeyInd)}{1} = Moment1Hat./( sqrt(Moment2Hat) + Epsilon );
        end
    end

    if strcmp(SharedWeights,'No')
        for KeyInd = 1:length(Keys)
            %Setup weight-specific number of layers
            if strcmp( Keys(KeyInd), 'W' )
                TWeight = 1;
            elseif strcmp( Keys(KeyInd), 'Dict' )
                TWeight = T+1;
            else
                TWeight = T;
            end

            %Modify
            for t = 1:TWeight
                Moment1{Keys(KeyInd)}{t} = Beta1*Moment1{Keys(KeyInd)}{t} + ( 1 - Beta1 )*Grads{Keys(KeyInd)}{t};
                Moment2{Keys(KeyInd)}{t} = Beta2*Moment2{Keys(KeyInd)}{t} + ( 1 - Beta2 )*( Grads{Keys(KeyInd)}{t}.^2 );
    
                Moment1Hat = Moment1{Keys(KeyInd)}{t}/( 1 - Beta1^Iterate );
                Moment2Hat = Moment2{Keys(KeyInd)}{t}/( 1 - Beta2^Iterate );
    
                Grads{Keys(KeyInd)}{t} = Moment1Hat./( sqrt(Moment2Hat) + Epsilon );
            end
        end
    end
end

%Update the weights
if strcmp(SharedWeights,'Yes')
    for KeyInd = 1:length(Keys)
        %Update
        NetWeights{Keys(KeyInd)}{1} = NetWeights{Keys(KeyInd)}{1} - LearningRate*Grads{Keys(KeyInd)}{1};

        %Weight-specific options
        if strcmp(Keys(KeyInd),'Dict')
            if strcmp(NormalizeDictionary,'Yes') %Normalize the dictionary
                NetWeights{Keys(KeyInd)}{1} = normalize(NetWeights{Keys(KeyInd)}{1},'norm',2);
            end
        end
        if strcmp(Keys(KeyInd),'b3A3')
            if strcmp(ProjectLastMLPWeights,'Yes') %Project the matrix
                NetWeights{Keys(KeyInd)}{1} = max( NetWeights{Keys(KeyInd)}{1}, Epsilon );
            end
        end

    end
end

if strcmp(SharedWeights,'No')
    for KeyInd = 1:length(Keys)
        %Setup weight-specific number of layers
        if strcmp(Keys(KeyInd), 'W')
            TWeight = 1;
        elseif strcmp(Keys(KeyInd), 'Dict')
            TWeight = T+1;
        else
            TWeight = T;
        end

        %Update
        for t = 1:TWeight
            NetWeights{Keys(KeyInd)}{t} = NetWeights{Keys(KeyInd)}{t} - LearningRate*Grads{Keys(KeyInd)}{t};
            
            %Weight-specific options
            if strcmp(Keys(KeyInd),'Dict')
                if strcmp(NormalizeDictionary,'Yes') %Normalize the dictionary
                    NetWeights{Keys(KeyInd)}{t} = normalize(NetWeights{Keys(KeyInd)}{t},'norm',2);
                end
            end
            if strcmp(Keys(KeyInd),'b3A3')
                if strcmp(ProjectLastMLPWeights,'Yes') %Project the matrix
                    NetWeights{Keys(KeyInd)}{t} = max( NetWeights{Keys(KeyInd)}{t}, Epsilon );
                end
            end

        end
    end
end

end

