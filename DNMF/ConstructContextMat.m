function ContextX = ConstructContextMat(X,m,N,T)
%CONSTRUCTCONTEXTMAT: contructs an enlarged version of M by concatenating T
%consecutive frames of M. 

ContextX = zeros(m*T,N);
WideM = [zeros(m,T-1), X];
for i = 1:T
   ContextX( 1+(i-1)*m : i*m ,:) = WideM(:, i : N+i-1 ); 
end
end

