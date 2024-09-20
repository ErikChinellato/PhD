function [NetWeights,Moment1,Moment2] = UpdateWeights(NetWeights,Grads,Moment1,Moment2,Dynamic,Iterate,GainMask,NetParameters)
%UPDATEWEIGHTS: Updates the net's weights.

%Variables
Layers = NetParameters.Layers;
SharedWeights = NetParameters.SharedWeights;
ProjectDynamics = NetParameters.ProjectDynamics;
GainLearningRate = NetParameters.GainLearningRate;
DynamicsLearningRate = NetParameters.DynamicsLearningRate;
Optimizer = NetParameters.Optimizer;
Epsilon = NetParameters.AdamEpsilon;

%Optimizer choice
if strcmp(Optimizer,'SGD')
    %Do nothing, use Grads
end

if strcmp(Optimizer,'Adam')
    %Modify Grads
    Beta1 = NetParameters.BetaMoment1;
    Beta2 = NetParameters.BetaMoment2;
    
    if strcmp(SharedWeights,'Yes')
        %Modify
        for Layer = 1:2
            if Layer < 2
                %Kalman Gains
                Moment1{Layer} = Beta1*Moment1{Layer} + ( 1 - Beta1 )*Grads{Layer};
                Moment2{Layer} = Beta2*Moment2{Layer} + ( 1 - Beta2 )*( Grads{Layer}.^2 );
        
                Moment1Hat = Moment1{Layer}/( 1 - Beta1^Iterate );
                Moment2Hat = Moment2{Layer}/( 1 - Beta2^Iterate );
        
                Grads{Layer} = Moment1Hat./( sqrt(Moment2Hat) + Epsilon );
            else
                %Dynamics
                Moment1{Layer}{Dynamic} = Beta1*Moment1{Layer}{Dynamic} + ( 1 - Beta1 )*Grads{Layer}{Dynamic};
                Moment2{Layer}{Dynamic} = Beta2*Moment2{Layer}{Dynamic} + ( 1 - Beta2 )*( Grads{Layer}{Dynamic}.^2 );
        
                Moment1Hat = Moment1{Layer}{Dynamic}/( 1 - Beta1^Iterate );
                Moment2Hat = Moment2{Layer}{Dynamic}/( 1 - Beta2^Iterate );
        
                Grads{Layer}{Dynamic} = Moment1Hat./( sqrt(Moment2Hat) + Epsilon );
            end
        end
    end

    if strcmp(SharedWeights,'No')
        %Modify
        for Layer = 1:Layers+1
            if Layer < Layers+1
                %Kalman Gains
                Moment1{Layer} = Beta1*Moment1{Layer} + ( 1 - Beta1 )*Grads{Layer};
                Moment2{Layer} = Beta2*Moment2{Layer} + ( 1 - Beta2 )*( Grads{Layer}.^2 );
    
                Moment1Hat = Moment1{Layer}/( 1 - Beta1^Iterate );
                Moment2Hat = Moment2{Layer}/( 1 - Beta2^Iterate );
    
                Grads{Layer} = Moment1Hat./( sqrt(Moment2Hat) + Epsilon );
            else
                %Dynamics
                Moment1{Layer}{Dynamic} = Beta1*Moment1{Layer}{Dynamic} + ( 1 - Beta1 )*Grads{Layer}{Dynamic};
                Moment2{Layer}{Dynamic} = Beta2*Moment2{Layer}{Dynamic} + ( 1 - Beta2 )*( Grads{Layer}{Dynamic}.^2 );
    
                Moment1Hat = Moment1{Layer}{Dynamic}/( 1 - Beta1^Iterate );
                Moment2Hat = Moment2{Layer}{Dynamic}/( 1 - Beta2^Iterate );
    
                Grads{Layer}{Dynamic} = Moment1Hat./( sqrt(Moment2Hat) + Epsilon );
            end
        end
    end
end

%Update the weights
if strcmp(SharedWeights,'Yes')
    %Update
    for Layer = 1:2
        if Layer < 2
            %Kalman Gains
            NetWeights{Layer} = NetWeights{Layer} - GainLearningRate*GainMask.*Grads{Layer};
        else
            %Dynamics
            NetWeights{Layer}{Dynamic} = NetWeights{Layer}{Dynamic} - DynamicsLearningRate*Grads{Layer}{Dynamic};

            if strcmp(ProjectDynamics,'Yes')
                %Project dynamics vector
                NetWeights{Layer}{Dynamic} = abs(NetWeights{Layer}{Dynamic});%max( NetWeights{Layer}{Dynamic}, Epsilon); %Alternatively: take the abs
            end
        end
    end
end

if strcmp(SharedWeights,'No')
    %Update
    for Layer = 1:Layers+1
        if Layer < Layers+1
            %Kalman Gains
            NetWeights{Layer} = NetWeights{Layer} - GainLearningRate*GainMask.*Grads{Layer};
        else
            %Dynamics
            NetWeights{Layer}{Dynamic} = NetWeights{Layer}{Dynamic} - DynamicsLearningRate*Grads{Layer}{Dynamic};

            if strcmp(ProjectDynamics,'Yes')
                %Project dynamics vector
                NetWeights{Layer}{Dynamic} = abs(NetWeights{Layer}{Dynamic});%max( NetWeights{Layer}{Dynamic}, Epsilon); %Alternatively: take the abs
            end
        end
    end
end
end


