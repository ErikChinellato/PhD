function y = SelectPatch(Y,m,n,k)
%SELECTPATCH: Selects the k-th patch y of size (m,n) from the image Y of size (M,N).
%The patch index k ranges from 1 to (M-m+1)*(N-n+1).

%Variables
[M,N] = size(Y);

%Check validity of k
if (k <= 0) || (k > (M-m+1)*(N-n+1))
    warning('The given patch index is negative or too big!');
    return;
end

%Select patch
RowStart = floor( (k-1)/(N-n+1) ) + 1;
ColStart = mod( (k-1), (N-n+1) ) + 1;

y = Y(RowStart:RowStart+m-1,ColStart:ColStart+n-1);
end

