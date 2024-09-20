function NetWeights = InitializeWeights(NetParameters)
%INITIALIZEWEIGHTS: Initializes the weights of the net. NetWeigths is a
%dictionary with keys 'Dict', 'C', 'W', 'b0A0', 'b1A1', 'b2A2' and 'b3A3'.
%associated to cells containing the corresponding weight(s).

%Variables
T = NetParameters.Layers;
PatchSize = NetParameters.PatchSize;
DictionarySize = NetParameters.DictionarySize;
MLPHS = NetParameters.MLPHiddenSizes;
SharedWeights = NetParameters.SharedWeights;

P = PatchSize^2;

%Initialize Dict
DictInit = OvercompleteDCT(PatchSize,DictionarySize);

%Initialize C
CInit = norm(DictInit,2)^2;

%Initialize W
WInit = normrnd(1,1/10,P,1);

%Initialize b0A0
b0A0Init = [zeros(MLPHS(1),1), normrnd(0,sqrt(2/MLPHS(1)),MLPHS(1),P)];

%Initialize b1A1
b1A1Init = [zeros(MLPHS(2),1), normrnd(0,sqrt(2/MLPHS(2)),MLPHS(2),MLPHS(1))];

%Initialize b2A2
b2A2Init = [zeros(MLPHS(3),1), normrnd(0,sqrt(2/MLPHS(3)),MLPHS(3),MLPHS(2))];

%Initialize b3A3
b3A3Init = [0, normrnd(0,sqrt(2),1,MLPHS(3))];

%Setup NetWeights
if strcmp(SharedWeights,'Yes')
    %Here the weights are shared between layers
    Dict = cell(1);
    C = cell(1);
    W = cell(1);
    b0A0 = cell(1);
    b1A1 = cell(1);
    b2A2 = cell(1);
    b3A3 = cell(1);

    W{1} = WInit;

    Dict{1} = DictInit;
    C{1} = CInit;
    b0A0{1} = b0A0Init;
    b1A1{1} = b1A1Init;
    b2A2{1} = b2A2Init;
    b3A3{1} = b3A3Init;
end

if strcmp(SharedWeights,'No')
    %Here the weights are NOT shared between layers
    Dict = cell(1,T+1);
    C = cell(1,T);
    W = cell(1);
    b0A0 = cell(1,T);
    b1A1 = cell(1,T);
    b2A2 = cell(1,T);
    b3A3 = cell(1,T);

    W{1} = WInit;

    for t = 1:T+1
        Dict{t} = DictInit;
        if t < T+1
            C{t} = CInit;
            b0A0{t} = b0A0Init;
            b1A1{t} = b1A1Init;
            b2A2{t} = b2A2Init;
            b3A3{t} = b3A3Init;
        end
    end

end

NetWeights = dictionary('Dict',{Dict}, 'C',{C}, 'W',{W}, 'b0A0',{b0A0}, 'b1A1',{b1A1}, 'b2A2',{b2A2}, 'b3A3',{b3A3});
end

