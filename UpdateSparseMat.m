function SparseMat = UpdateSparseMat(NetWeights,States,ModelDiscoverySupport,Dynamic,NetParameters)
%UPDATESPARSEMAT: Updates the sparse matrix for model discovery.

%Variables
Experiment = NetParameters.Experiment;
Layers = NetParameters.Layers;
StateDimension = NetParameters.StateDimension;
Model = NetParameters.Model;
FirstStateOffset = NetParameters.ModelDiscoveryFirstState;
ModelDiscoveryMethod = NetParameters.ModelDiscoveryMethod;
ModelDiscoverySmoothing = NetParameters.ModelDiscoverySmoothing;

%Assemble X matrix & create weight vector
X = zeros(StateDimension,Layers-FirstStateOffset);
WeightVec = ones(Layers-FirstStateOffset,1);
for Layer = 1:Layers-FirstStateOffset
    X(:,Layer) = States{Layer+FirstStateOffset};
    Sigma = svds(NetWeights{Layer+FirstStateOffset},1);
    %WeightVec(Layer) = 1/Sigma;
end

%Compute X' matrix and possibly a smoothed version of X
if strcmp(ModelDiscoverySmoothing,'TV')
    %Smoother-specific variables
    A = NetParameters.A;
    D = NetParameters.D;
    AtA = NetParameters.AtA;
    B = NetParameters.B;

    %Smoother
    [XPrimeTarget,XTarget] = ComputeTVDerivative(X,A,D,AtA,B,Model.SamplingTimes(1)); 
    XPrimeTarget = smoothdata(XPrimeTarget,2,'sgolay');
end
if strcmp(ModelDiscoverySmoothing,'TVMixed')
    %Smoother-specific variables
    A = NetParameters.A;
    D = NetParameters.D;
    AtA = NetParameters.AtA;
    B = NetParameters.B;

    %Smoother
    [~,XTarget] = ComputeTVDerivative(X,A,D,AtA,B,Model.SamplingTimes(1));
    XPrimeTarget = ( XTarget(:,3:end) - XTarget(:,1:end-2) )./(2*Model.SamplingTimes(1)');
    XTarget = XTarget(:,2:end-1); %We lose two time instants due to central finite differences
    WeightVec = WeightVec(2:end-1);
end
if strcmp(ModelDiscoverySmoothing,'TVMixed2')
    %Smoother-specific variables
    A = NetParameters.A;
    D = NetParameters.D;
    AtA = NetParameters.AtA;
    B = NetParameters.B;

    %Smoother
    XPrimeTarget = ComputeTVDerivative(X,A,D,AtA,B,Model.SamplingTimes(1));
    XTarget = X;
end
if strcmp(ModelDiscoverySmoothing,'SG')
    %Smoother-specific variables
    StencilA0 = NetParameters.StencilA0;
    StencilA1 = NetParameters.StencilA1;
    WinLen = NetParameters.WinLen;

    %Smoother
    [XPrimeTarget,XTarget] = ComputeSGDerivative(X,StencilA0,StencilA1,WinLen,Model.SamplingTimes(1));
end
if strcmp(ModelDiscoverySmoothing,'SGMixed')
    %Smoother-specific variables
    StencilA0 = NetParameters.StencilA0;
    StencilA1 = NetParameters.StencilA1;
    WinLen = NetParameters.WinLen;

    %Smoother
    [~,XTarget] = ComputeSGDerivative(X,StencilA0,StencilA1,WinLen,Model.SamplingTimes(1));
    [XPrimeTarget,XTarget] = ComputeSGDerivative(XTarget,StencilA0,StencilA1,WinLen,Model.SamplingTimes(1));
    %XTarget = smoothdata(XTarget,2,'sgolay'); %HERE
    %XPrimeTarget = ( XTarget(:,3:end) - XTarget(:,1:end-2) )./(2*Model.SamplingTimes(1)');
    %XTarget = XTarget(:,2:end-1); %We lose two time instants due to central finite differences
    %XPrimeTarget = smoothdata(XPrimeTarget,2,'gaussian'); %HERE
    %[~,XPrimeTarget] = ComputeSGDerivative(XTarget,StencilA0,StencilA1,WinLen,Model.SamplingTimes(1));
    %WeightVec = WeightVec(2:end-1);
end
if strcmp(ModelDiscoverySmoothing,'SGMixed2')
    %Smoother-specific variables
    StencilA0 = NetParameters.StencilA0;
    StencilA1 = NetParameters.StencilA1;
    WinLen = NetParameters.WinLen;

    %Smoother
    XPrimeTarget = ComputeSGDerivative(X,StencilA0,StencilA1,WinLen,Model.SamplingTimes(1));
    XTarget = X;
end
if strcmp(ModelDiscoverySmoothing,'SGMixed3')
    %Smoother
    XTarget = smoothdata(X,2,'sgolay');
    XPrimeTarget = ( XTarget(:,3:end) - XTarget(:,1:end-2) )./(2*Model.SamplingTimes(1)');
    XTarget = XTarget(:,2:end-1); %We lose two time instants due to central finite differences
    %XPrimeTarget = smoothdata(XPrimeTarget,2,'gaussian'); %HERE
    WeightVec = WeightVec(2:end-1);
end
if strcmp(ModelDiscoverySmoothing,'No')
    XPrimeTarget = ( X(:,3:end) - X(:,1:end-2) )./(2*Model.SamplingTimes(1)');
    XTarget = X(:,2:end-1); %We lose two time instants due to central finite differences
    WeightVec = WeightVec(2:end-1);

    %XPrimeTarget  = (X(:,2:end)-X(:,1:end-1))./Model.SamplingTimes(1)';
    %XTarget = X(:,1:end-1); %We lose one time instant due to forward finite differences
end

ColNum = size(XPrimeTarget,2);

%Assemble target matrix
if strcmp(Experiment,'1')
    Target = ( Model.M*XPrimeTarget - Model.K*XTarget - Model.D(:,1:ColNum) )';
end
if strcmp(Experiment,'2')
    Target = ( Model.M*XPrimeTarget - Model.K*XTarget - Model.D(:,1:ColNum) )';
end
if strcmp(Experiment,'3')
    Target = ( Model.M*XPrimeTarget - Model.K*XTarget - [zeros(1,ColNum);-XTarget(1,:).*XTarget(3,:);XTarget(1,:).*XTarget(2,:)] - Model.D(:,1:ColNum) )';
end
if strcmp(Experiment,'4')
    Target = ( ( Model.M + diag([NetWeights{end}{Dynamic},zeros(1,StateDimension-1)]) )*XPrimeTarget - Model.K*XTarget - [zeros(1,ColNum);-XTarget(1,:).*XTarget(3,:);XTarget(1,:).*XTarget(2,:)] - Model.D(:,1:ColNum) )';
end
if strcmp(Experiment,'5')
    Target = ( Model.M*XPrimeTarget - ( Model.K + [[0,0,0];[NetWeights{end}{Dynamic},0,0];[0,0,0]] )*XTarget - [zeros(1,ColNum);-XTarget(1,:).*XTarget(3,:);XTarget(1,:).*XTarget(2,:)] - Model.D(:,1:ColNum) )';
end
if strcmp(Experiment,'6')
    Target = ( Model.M*XPrimeTarget - Model.K*XTarget - Model.D(:,1:ColNum) )';
end

%Smoother
%Target = smoothdata(Target,'sgolay');
Target = WeightVec(1:ColNum).*Target;

%Construct dictionary & normalize it
Phi = WeightVec(1:ColNum).*ConstructDictionary(XTarget,NetParameters);
Norms = vecnorm(Phi);
Phi = Phi./Norms;

%Update the sparse matrix with sparse coder
if strcmp(ModelDiscoveryMethod,'OMP')
    %OMP variables
    OMPSparsity = NetParameters.OMPSparsity;
    ModelDiscoveryRelativeThreshold = NetParameters.ModelDiscoveryRelativeThreshold;

    %OMP
    SparseMat = zeros( size(NetWeights{end}{end}) );
    for State = 1:StateDimension
        if any(ModelDiscoverySupport(:,State))
            Temp = OMP(OMPSparsity,Target(:,State)',Phi(:,ModelDiscoverySupport(:,State)));
            
            %Pruning
            RelRes = norm(Phi(:,ModelDiscoverySupport(:,State))*Temp'-Target(:,State))/norm(Target(:,State));
            ContinuePruning = 1;

            while ContinuePruning 
                Support = find(Temp);
                PruneRelRes = inf;
    
                for SuppIndx = Support
                    TempVal = Temp;
                    TempVal(SuppIndx) = 0;
    
                    RelResVal = norm(Phi(:,ModelDiscoverySupport(:,State))*TempVal'-Target(:,State))/norm(Target(:,State));
                    
                    if RelResVal < PruneRelRes
                        PruneRelRes = RelResVal;
                        PruneIndx = SuppIndx;
                    end
                end
    
                if PruneRelRes < (1 + ModelDiscoveryRelativeThreshold)*RelRes
                    Temp(PruneIndx) = 0;
                else
                    ContinuePruning = 0;
                end
            end

            %Old pruning
            %Temp( abs(Temp) < ModelDiscoveryRelativeThreshold*norm(Target(:,State)) ) = 0;

            SparseMat(ModelDiscoverySupport(:,State),State) = Temp;
        end
    end
end
if strcmp(ModelDiscoveryMethod,'LH')
    %LH variables
    ModelDiscoveryRelativeThreshold = NetParameters.ModelDiscoveryRelativeThreshold;

    %LH with positivity trick
    SparseMat = zeros( size(NetWeights{end}{end}) );
    for State = 1:StateDimension
        if any(ModelDiscoverySupport(:,State))
            TempPos = lsqnonneg(Phi(:,ModelDiscoverySupport(:,State)),Target(:,State));
            TempNeg = lsqnonneg(-Phi(:,ModelDiscoverySupport(:,State)),Target(:,State));
            Temp = TempPos-TempNeg;

            %Pruning
            RelRes = norm(Phi(:,ModelDiscoverySupport(:,State))*Temp'-Target(:,State))/norm(Target(:,State));
            ContinuePruning = 1;

            while ContinuePruning 
                Support = find(Temp);
                PruneRelRes = inf;
    
                for SuppIndx = Support
                    TempVal = Temp;
                    TempVal(SuppIndx) = 0;
    
                    RelResVal = norm(Phi(:,ModelDiscoverySupport(:,State))*TempVal'-Target(:,State))/norm(Target(:,State));
                    
                    if RelResVal < PruneRelRes
                        PruneRelRes = RelResVal;
                        PruneIndx = SuppIndx;
                    end
                end
    
                if PruneRelRes < (1 + ModelDiscoveryRelativeThreshold)*RelRes
                    Temp(PruneIndx) = 0;
                else
                    ContinuePruning = 0;
                end
            end

            %Old pruning
            %Temp( abs(Temp) < ModelDiscoveryRelativeThreshold*norm(Target(:,State)) ) = 0;

            SparseMat(ModelDiscoverySupport(:,State),State) = Temp;
        end
    end
end
if strcmp(ModelDiscoveryMethod,'ISTA')
    %ISTA Variables
    ISTAThreshold = NetParameters.ISTAThreshold;
    ISTAMaxIt = NetParameters.ISTAMaxIt;

    %Compute spectral norm squared of the dictionary and some other useful matrices
    DtD = Phi'*Phi;
    C = abs(eigs(DtD,1));
    DtDdC = DtD/C;
    DtTargetdC = (Phi'*Target)/C;
    Thresh = ISTAThreshold/C;

    %ISTA
    SparseMat = zeros( size(NetWeights{end}{end}) );
    %SparseMat = NetWeights{end}{end};
    SparseMat = ISTA(DtDdC,DtTargetdC,SparseMat,Thresh,ISTAMaxIt);
end

%Recover sparse coefficients
SparseMat = SparseMat./Norms';

end
