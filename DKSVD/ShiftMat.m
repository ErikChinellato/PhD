function A = ShiftMat(A,shift)
%SHIFTMAT: cyclic shift on the right of the columns of A by shift.

A = [A(:,end-shift+1:end),A(:,1:end-shift)];
end

