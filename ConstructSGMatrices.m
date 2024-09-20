function [StencilA0,StencilA1] = ConstructSGMatrices(WinLen)
%CONSTRUCTSGMATRICES: Construct matrices used during SG filtering.

%Variables
HalfWinLen = (WinLen-1)/2;
Int = -HalfWinLen:HalfWinLen;

%Degree = 3; %FIXED FOR NOW!
StencilA0 = flip( ( 3/( 4*WinLen*(WinLen^2-4) ) )*( 3*WinLen^2 - 7 - 20*Int.^2 ) );
StencilA1 = flip( ( 1/( WinLen*(WinLen^2-1)*(3*WinLen^4-39*WinLen^2+108) ) )*( 75*( 3*WinLen^4 - 18*WinLen^2 + 31 )*Int - 420*( 3*WinLen^2 - 7 )*Int.^3 ) );

end

