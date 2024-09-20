function y = R(Y,p,k)
%R: Selects the k-th square patch y of size (p,p) from the (square) image Y of size (M,N). 
%Y is a matrix, while y is the vectorized (row-wise) patch. The patch index
%k ranges from 1 to (M-p+1)*(N-p+1).

%Variables
[M,N] = size(Y);

%Check validity of k
if (k <= 0) || (k > (M-p+1)*(N-p+1))
    warning('The given patch index is negative or too big!');
    return;
end

%Select patch
RowStart = floor( (k-1)/(N-p+1) ) + 1;
ColStart = mod( (k-1), (N-p+1) ) + 1;

y = Y(RowStart:RowStart+p-1,ColStart:ColStart+p-1);
y = reshape(y',p^2,1);
end

