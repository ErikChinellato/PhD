function [XPrime,XSmooth] = ComputeSGDerivative(X,StencilA0,StencilA1,WinLen,TimeStep)
%COMPUTESGDERIVATIVE: Computes the matrix X' = XPrime and XSmooth using SG filter.

%Variables 
HalfWinLen = (WinLen-1)/2;

XPrime = zeros(size(X));
if nargout > 1
    XSmooth = zeros(size(X));
end

States = size(X,1);

for State = 1:States
    %Select current state time series
    CurrState = X(State,:)';

    %Extend it for polynomial fits at the boundaries
    %CurrStateExt = [ flip(CurrState(2:2+HalfWinLen-1)); CurrState; flip(CurrState(end-HalfWinLen:end-1)) ];
    %CurrStateExt = [ CurrState(1)*ones(HalfWinLen,1); CurrState; CurrState(end)*ones(HalfWinLen,1)];
    CurrStateExt = [ -flip(CurrState(2:2+HalfWinLen-1))+2*CurrState(1); CurrState; -flip(CurrState(end-HalfWinLen:end-1))+2*CurrState(end)];
   
    %Compute coefficients of fitted polynomials (only first 2 are needed)
    A0 = conv(CurrStateExt,StencilA0,'valid');
    A1 = conv(CurrStateExt,StencilA1,'valid');
    
    XPrime(State,:) = (A1/TimeStep)';
    if nargout > 1
        XSmooth(State,:) = A0';
    end
end

