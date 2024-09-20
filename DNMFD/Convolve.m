function ConvAB = Convolve(A,B)
%CONVOLVE: Convolves the two matrices A and B. If size(B,1) = 1 we convolve 
%forward in time using the columns of A and obtain a matrix of size (size(A,1),size(B,2)), 
%otherwise backward in time using the transpose of the columns of A and obtain 
%a matrix of size (1,size(B,2)).

if size(B,1) == 1
    ConvAB = zeros(size(A,1),size(B,2));
    for ColInd = 1:size(A,2)
        ConvAB = ConvAB + A(:,ColInd)*ShiftMat(B,-(ColInd-1));
    end
else
    ConvAB = zeros(1,size(B,2));
    for ColInd = 1:size(A,2)
        ConvAB = ConvAB + A(:,ColInd)'*ShiftMat(B,ColInd-1);
    end
end

end

