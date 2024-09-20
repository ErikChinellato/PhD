function [PosGradH, NegGradH, PosGradW, NegGradW] = ComputeLastGrad(X,W,H,Source,epsilon,InitializationRanks,FactRanks,Objective)
%COMPUTELASTGRAD: computes the positive and negative parts of the gradients
%for W and H at the last layer. 

%% Common components between objectives
SourceRank = FactRanks(1);
WSource = W(:,1:SourceRank);
WSourceCompl = W(:,SourceRank+1:end);
HSource = H(1:SourceRank,:);
HSourceCompl = H(SourceRank+1:end,:);

Lam = W*H;
LamSource = WSource*HSource;
%LamSourceCompl = Lam - LamSource;
LamSourceCompl = WSourceCompl*HSourceCompl;

%% Main objective choice options
%WienerFilter reconstruction objective
if strcmp(Objective,'WienerFilter')

%LamSq = Lam.^2;
%XdLamSq = X./(LamSq+eps);
%XSdLamSq = XdLamSq.*Source;
%XLamSdLam = X.*(LamSource./(Lam+eps));

%PosSource = XdLamSq.*XLamSdLam.*LamSourceCompl;
%NegSourceCompl = XdLamSq.*XLamSdLam.*LamSource;

%PosSourceCompl = XSdLamSq.*LamSource;
%NegSource = XSdLamSq.*LamSourceCompl;

PosSource = ((X.^2).*LamSource.*LamSourceCompl)./(Lam.^3+epsilon);
NegSource = (X.*Source.*LamSourceCompl)./(Lam.^2+epsilon);
PosSourceCompl = (X.*Source.*LamSource)./(Lam.^2+epsilon);
NegSourceCompl = ((X.^2).*(LamSource.^2))./(Lam.^3+epsilon);

PosGradW = [PosSource*HSource', PosSourceCompl*HSourceCompl'];
NegGradW = [NegSource*HSource', NegSourceCompl*HSourceCompl'];

PosGradH = [WSource'*PosSource; WSourceCompl'*PosSourceCompl];
NegGradH = [WSource'*NegSource; WSourceCompl'*NegSourceCompl];
end

%WienerFilterReconstruction reconstruction objective
if strcmp(Objective,'WienerFilterReconstruction')
ReconPen = 1e40; 
SparsePen = 0;

LamSq = Lam.^2;
XdLamSq = X./(LamSq+epsilon);
XSdLamSq = XdLamSq.*Source;
XLamSdLam = X.*(LamSource./(Lam+epsilon));

PosSource = XdLamSq.*XLamSdLam.*LamSourceCompl+ReconPen*LamSource;
NegSourceCompl = XdLamSq.*XLamSdLam.*LamSource;

PosSourceCompl = XSdLamSq.*LamSource;
NegSource = (XSdLamSq.*LamSourceCompl)+ReconPen*Source;

PosGradW = [PosSource*HSource', PosSourceCompl*HSourceCompl'];
NegGradW = [NegSource*HSource', NegSourceCompl*HSourceCompl'];

PosGradH = [WSource'*PosSource+SparsePen; WSourceCompl'*PosSourceCompl];
NegGradH = [WSource'*NegSource; WSourceCompl'*NegSourceCompl];
end

%% Other objective choice options
%Weighted WienerFilter reconstruction objective
if strcmp(Objective,'WeightedWienerFilter')

XLamS = X.*LamSource;
XLamSSq = XLamS.^2;
LamLamSC = Lam.*LamSourceCompl;
LamLamSCSq = LamLamSC.^2;
LamSCSq = LamSourceCompl.^2;
XLamSS = XLamS.*Source;


PosSource = ((Source+X).*XLamS)./(LamLamSCSq+epsilon);
NegSource = XLamSSq./(Lam.*LamLamSCSq+epsilon) + (Source.*X)./(LamLamSC.*LamSourceCompl+epsilon);

PosSourceCompl = 2*XLamSS./(LamLamSC.*LamSCSq+epsilon) + XLamSS./(LamLamSCSq+epsilon);
NegSourceCompl = (Source.^2)./(LamSCSq.*LamSourceCompl+epsilon) + XLamSSq./(LamLamSCSq.*LamSourceCompl+epsilon) + XLamSSq./(LamLamSCSq.*Lam+epsilon);
    
PosGradH = [WSource'*PosSource; WSourceCompl'*PosSourceCompl];
NegGradH = [WSource'*NegSource; WSourceCompl'*NegSourceCompl];    
end

%SNRMask reconstruction objective
if strcmp(Objective,'SNRMask')

LamSCSq = LamSourceCompl.^2;
LamSCCb = LamSCSq.*LamSourceCompl;

PosSource = LamSource./(LamSCSq+epsilon);
NegSource = Source./(LamSCSq+epsilon);

PosSourceCompl = (2*Source.*LamSource)./(LamSCCb+epsilon);
NegSourceCompl = (Source.^2+LamSource.^2)./(LamSCCb+epsilon);

PosGradH = [WSource'*PosSource; WSourceCompl'*PosSourceCompl];
NegGradH = [WSource'*NegSource; WSourceCompl'*NegSourceCompl];
end

%WienerFilterReconstruction reconstruction objective
if strcmp(Objective,'WienerFilterSparseReconstruction')
ReconPen = 1e200; 
SparsePen = 1e300;

SparseMat = ones(size(H));
for BlockInd = BlockDictInds
    SparseMat(1+sum(InitializationRanks{1}(1:BlockInd-1)):sum(InitializationRanks{1}(1:BlockInd)),:) = 0;
end
SparseMat(SourceRank+1:end,:) = 0;

LamSq = Lam.^2;
XdLamSq = X./(LamSq+epsilon);
XSdLamSq = XdLamSq.*Source;
XLamSdLam = X.*(LamSource./(Lam+epsilon));

PosSource = XdLamSq.*XLamSdLam.*LamSourceCompl+ReconPen*LamSource;
NegSourceCompl = XdLamSq.*XLamSdLam.*LamSource;

PosSourceCompl = XSdLamSq.*LamSource;
NegSource = (XSdLamSq.*LamSourceCompl)+ReconPen*Source;

%PosGradW = [PosSource*HSource',PosSourceCompl*HSourceCompl'];
%NegGradW = [NegSource*HSource',NegSourceCompl*HSourceCompl'];

PosGradH = [WSource'*PosSource; WSourceCompl'*PosSourceCompl]+SparsePen*SparseMat;
NegGradH = [WSource'*NegSource; WSourceCompl'*NegSourceCompl];
end

%PureReconstruction objective
if strcmp(Objective,'PureReconstruction')

PosSource = LamSource;
NegSourceCompl = (1e-20)*ones(size(WSourceCompl'*LamSource));

PosSourceCompl = (1e-20)*ones(size(WSourceCompl'*LamSource));
NegSource = Source;

%PosGradW = [PosSource*HSource',PosSourceCompl*HSourceCompl'];
%NegGradW = [NegSource*HSource',NegSourceCompl*HSourceCompl'];

PosGradH = [WSource'*PosSource; PosSourceCompl];
NegGradH = [WSource'*NegSource; NegSourceCompl];
end

end
