function AShift = ShiftMat(A,Shift)
%SHIFTMAT: Shifts the matrix A by Shift columns. The dimension in preserved and
%the extra columns are filled with zeros. If Shift >= 0 the columns are shifted
%to the left (backward) while if Shift < 0 the columns are shifted to the right (forward).

AShift = zeros(size(A));

if Shift >= 0
    AShift(:,1:end-Shift) = A(:,Shift+1:end);
else
    AShift(:,-Shift+1:end) = A(:,1:end+Shift);
end

end

