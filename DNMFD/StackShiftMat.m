function B = StackShiftMat(A,T)
%STACKSHIFTMAT: Vertically stacks some shifted versions (T times) of the
%matrix A, then takes the transpose.

%Set up output matrix
B = zeros(T,size(A,2));

for RowInd = 1:T
    B(RowInd,:) = ShiftMat(A,-(RowInd-1));
end

B = B';

end

