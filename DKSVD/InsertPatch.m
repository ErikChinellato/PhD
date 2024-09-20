function Y = InsertPatch(y,M,N,k)
%INSERTPATCH: Inserts the patch y of size (m,n) in the k-th patch slot
%of an image Y of size (M,N).

%Variables
[m,n] = size(y);

%Check validity of k
if (k <= 0) || (k > (M-m+1)*(N-n+1))
    warning('The given patch index is negative or too big!');
    return;
end

%Insert patch
Y = zeros(M,N);

RowStart = floor( (k-1)/(N-n+1) ) + 1;
ColStart = mod( (k-1), (N-n+1) ) + 1;

Y(RowStart:RowStart+m-1,ColStart:ColStart+n-1) = y;
end

