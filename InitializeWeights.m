function NetWeights = InitializeWeights(NetParameters)
%INITIALIZEWEIGHTS: Initializes the net's weights with gaussian noise of
%mean NetParameters.InitializationMean and sigma NetParameters.InitializationSigma.

%Variables
Experiment = NetParameters.Experiment;
Layers = NetParameters.Layers;
SharedWeights = NetParameters.SharedWeights;
Initialization = NetParameters.Initialization;
StateDimension = NetParameters.StateDimension;
ObservationDimension = NetParameters.ObservationDimension;
HiddenDynamicsNumber = NetParameters.HiddenDynamicsNumber;
HiddenDynamicsDimension = NetParameters.HiddenDynamicsDimension;
DictionaryDimension = NetParameters.DictionaryDimension;

C = NetParameters.C;
Model = NetParameters.Model;

%Setup weights
if strcmp(SharedWeights,'Yes')
    NetWeights = cell(1,2); 
    
    if strcmp(Initialization,'Deterministic')
        %Deterministic initialization
        if strcmp(Experiment,'1')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.SamplingTimes(1)^2*Model.AInit*Model.QInit*Model.AInit';
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            NetWeights{1} = KFGain;
        end

        if strcmp(Experiment,'2')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            NetWeights{1} = KFGain;
        end

        if strcmp(Experiment,'3')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            NetWeights{1} = KFGain;
        end

        if strcmp(Experiment,'4')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            NetWeights{1} = KFGain;
        end

        if strcmp(Experiment,'5')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            NetWeights{1} = KFGain;
        end

        if strcmp(Experiment,'6')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            NetWeights{1} = KFGain;
        end

        for Dyn = 1:HiddenDynamicsNumber
            NetWeights{2}{Dyn} = InitializeDynamics(HiddenDynamicsDimension(Dyn),Model,Experiment);
        end
        NetWeights{2}{HiddenDynamicsNumber+1} = InitializeSparseDynamicsMat(DictionaryDimension,StateDimension,Model,Experiment);
    end

    if strcmp(Initialization,'Random')
        %Random initialization
        Mean = NetParameters.InitializationMean;
        Sigma = NetParameters.InitializationSigma;

        NetWeights{1} = normrnd(Mean,Sigma,ObservationDimension,ObservationDimension);
        for Dyn = 1:HiddenDynamicsNumber
            NetWeights{2}{Dyn} = normrnd(Mean,Sigma,HiddenDynamicsDimension(Dyn),1);
        end
        NetWeights{2}{HiddenDynamicsNumber+1} = normrnd(Mean,Sigma,DictionaryDimension,StateDimension);
    end
end

if strcmp(SharedWeights,'No')
    NetWeights = cell(1,Layers+1);
    
    if strcmp(Initialization,'Deterministic')
        %Deterministic initialization
        if strcmp(Experiment,'1')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.SamplingTimes(1)^2*Model.AInit*Model.QInit*Model.AInit';
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            for Layer = 1:Layers
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'2')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            for Layer = 1:Layers
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'3')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            for Layer = 1:Layers
                NetWeights{Layer} = KFGain;%1e-1*eye(3);%KFGain;
            end
        end

        if strcmp(Experiment,'4')
            P = Model.PInit;
            A = Model.AInit;
            Q = 0;%Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            for Layer = 1:Layers
                NetWeights{Layer} = KFGain;%1e-1*eye(3);%KFGain;
            end
        end

        if strcmp(Experiment,'5')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            for Layer = 1:Layers
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'6')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;

            InvP = inv(A * P * A' + Q) + C'*InvR*C;   
            KFGain = InvP\(C'*InvR); %P*C'*InvR;

            for Layer = 1:Layers
                NetWeights{Layer} = KFGain;
            end
        end

        for Dyn = 1:HiddenDynamicsNumber
            NetWeights{Layers+1}{Dyn} = InitializeDynamics(HiddenDynamicsDimension(Dyn),Model,Experiment);
        end
        NetWeights{Layers+1}{HiddenDynamicsNumber+1} = InitializeSparseDynamicsMat(DictionaryDimension,StateDimension,Model,Experiment);
    end

    if strcmp(Initialization,'DeterministicComplete')
        %Deterministic complete initialization
        if strcmp(Experiment,'1')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.SamplingTimes(1)^2*Model.AInit*Model.QInit*Model.AInit';
            InvR = Model.invRInit;
            for Layer = 1:Layers
                InvP = inv(A * P * A' + Q) + C'*InvR*C;
                P = inv(InvP);    
                KFGain = InvP\(C'*InvR); %P*C'*InvR;
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'2')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;
            for Layer = 1:Layers
                InvP = inv(A * P * A' + Q) + C'*InvR*C;
                P = inv(InvP);    
                KFGain = InvP\(C'*InvR); %P*C'*InvR;
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'3')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;
            for Layer = 1:Layers
                InvP = inv(A * P * A' + Q) + C'*InvR*C;
                P = inv(InvP);    
                KFGain = InvP\(C'*InvR); %P*C'*InvR;
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'4')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;
            for Layer = 1:Layers
                InvP = inv(A * P * A' + Q) + C'*InvR*C;
                P = inv(InvP);    
                KFGain = InvP\(C'*InvR); %P*C'*InvR;
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'5')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;
            for Layer = 1:Layers
                InvP = inv(A * P * A' + Q) + C'*InvR*C;
                P = inv(InvP);    
                KFGain = InvP\(C'*InvR); %P*C'*InvR;
                NetWeights{Layer} = KFGain;
            end
        end

        if strcmp(Experiment,'6')
            P = Model.PInit;
            A = Model.AInit;
            Q = Model.QInit;
            InvR = Model.invRInit;
            for Layer = 1:Layers
                InvP = inv(A * P * A' + Q) + C'*InvR*C;
                P = inv(InvP);    
                KFGain = InvP\(C'*InvR); %P*C'*InvR;
                NetWeights{Layer} = KFGain;
            end
        end

        for Dyn = 1:HiddenDynamicsNumber
            NetWeights{Layers+1}{Dyn} = InitializeDynamics(HiddenDynamicsDimension(Dyn),Model,Experiment);
        end
        NetWeights{Layers+1}{HiddenDynamicsNumber+1} = InitializeSparseDynamicsMat(DictionaryDimension,StateDimension,Model,Experiment);
    end

    if strcmp(Initialization,'Random')
        %Random initialization
        Mean = NetParameters.InitializationMean;
        Sigma = NetParameters.InitializationSigma;

        for Layer = 1:Layers
            NetWeights{Layer} = normrnd(Mean,Sigma,ObservationDimension,ObservationDimension);
        end
        for Dyn = 1:HiddenDynamicsNumber
            NetWeights{Layers+1}{Dyn} = normrnd(Mean,Sigma,HiddenDynamicsDimension(Dyn),1);
        end
        NetWeights{Layers+1}{HiddenDynamicsNumber+1} = InitializeSparseDynamicsMat(DictionaryDimension,StateDimension,Model,Experiment);
    end
end

end

