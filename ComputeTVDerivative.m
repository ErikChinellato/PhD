function [XPrime,XSmooth] = ComputeTVDerivative(X,A,D,AtA,B,TimeStep)
%COMPUTETVDERIVATIVE: Computes the matrix X' = XPrime and XSmooth using TV regularizer.

%Variables
AlphaNum = 15;
AlphaMin = 1e-10;
AlphaMax = 0.5;
AlphaInterval = logspace(log10(AlphaMin),log10(AlphaMax),AlphaNum);
SearchMaxIt = 100;
FinalMaxIt = 1000;

XPrime = zeros(size(X));
if nargout > 1
    XSmooth = zeros(size(X));
end

States = size(X,1);

for State = 1:States
    %Select current state time series
    CurrState = X(State,:)';

    %Normalize it
    %Offset = CurrState(1);
    %CurrStateNorm = CurrState(1:end-1) - Offset;
    CurrStateNorm = ( CurrState(1:end-1) + CurrState(2:end))/2;
    Offset2 = CurrStateNorm(1);
    CurrStateNorm = CurrStateNorm - Offset2;

    AtCurrState = A'*CurrStateNorm;
    uInit = (1/TimeStep)*[0;diff(CurrStateNorm);0];

    %Estimate best alpha based on whiteness
    PeriodogramResidueOpt = inf;
    
    for Alpha = AlphaInterval
        %Estimate denoised state time series for current alpha value
        [~,CurrStateEst] = TVDifferentiate2(TimeStep,Alpha,B,D,AtA,AtCurrState,uInit,SearchMaxIt); 
        CurrStateEst = CurrStateEst + Offset2;
    
        %Compute cumulative periodogram of the error & residue wrt white noise
        CumulativePeriodogram = TestBartlett( CurrState(2:end) - CurrStateEst);
        PeriodogramResidue = norm(CumulativePeriodogram'-linspace(0,1,length(CumulativePeriodogram)));
    
        %If current value of alpha creates a whiter residue, swap it
        if PeriodogramResidue < PeriodogramResidueOpt
            PeriodogramResidueOpt = PeriodogramResidue;
            AlphaOpt = Alpha;
        end
    end
    
    %Estimate derivative for optimal alpha value
    [CurrStateDerivativeEst,CurrStateEst] = TVDifferentiate2(TimeStep,AlphaOpt,B,D,AtA,AtCurrState,uInit,FinalMaxIt); 
    XPrime(State,:) = CurrStateDerivativeEst';
    if nargout > 1
        XSmooth(State,:) = ([0;CurrStateEst] + Offset2)';
    end
end

end

