function Rks = BuildRks(N,P)
%BUILDRKS: returns a cell array of size ( 1, (sqrt(N)-sqrt(P)+1)^2 )
%containing the Rk matrices.

sqN = int32(sqrt(N));
sqP = int32(sqrt(P));

%Constructing common block to be shifted
A = zeros(P,P+(sqP-1)*(sqN-sqP));
for BlockInd = 1:sqP
   A( (BlockInd-1)*sqP+1:(BlockInd-1)*sqP+sqP, (BlockInd-1)*sqN+1:(BlockInd-1)*sqN+sqP) = eye(sqP); 
end

%Initializing the cell array
Rks = cell(1,(sqN-sqP+1)^2+1);
Rks{1} = sparse( [A,zeros(P,N-P-(sqP-1)*(sqN-sqP))] );

%Beginning the shifting process
shift = 0;
for RowInd = 1:sqN-sqP+1
   for ColInd = 1:sqN-sqP+1
       Rks{(RowInd-1)*(sqN-sqP+1)+ColInd+1} = sparse( ShiftMat(Rks{(RowInd-1)*(sqN-sqP+1)+ColInd},shift) );
       if ColInd == 1
          shift = 1;    %Beginning of row, next iterates shift by 1
       end
   end
   shift = sqP; %End of row, next iterate shift by sqrt(P)
end

%Removing the first element since it is a copy of the second
Rks = Rks(2:end);
end

