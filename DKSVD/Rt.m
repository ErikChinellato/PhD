function Y = Rt(y,Size,k)
%RT: Inserts the k-th square patch y of size (p,p) into the image Y of size Size. 
%Y is a matrix, while y is the vectorized (row-wise) patch. The patch index
%k ranges from 1 to (Size(1)-p+1)*(Size(2)-p+1).

%Variables
p = sqrt(length(y));

%Check validity of k
if (k <= 0) || (k > (Size(1)-p+1)*(Size(2)-p+1))
    warning('The given patch index is negative or too big!');
    return;
end

%Insert patch
Y = zeros(Size);

RowStart = floor( (k-1)/(Size(2)-p+1) ) + 1;
ColStart = mod( (k-1), (Size(2)-p+1) ) + 1;

Y(RowStart:RowStart+p-1,ColStart:ColStart+p-1) = reshape(y,p,p)';
end

