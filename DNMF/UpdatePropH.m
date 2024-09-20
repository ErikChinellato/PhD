function HNew = UpdatePropH(X,W,HCurr,epsilon,SparsePen,N)
%UPDATEH: updates the coefficient matrix HNew = H_{k+1} given the last iterate
%HCurr = H_{k} and the weight W_{k} for the (last C) discriminative layers 
%of the net. The update rule is that for sparse NMF with beta = 1.

%Num = W'*(X./(max(W*HCurr,epsilon))); 
Num = W'*(X./(W*HCurr+epsilon));  
Den = repmat(sum(W)',1,N) + SparsePen; 

HNew = max(epsilon, HCurr.*(Num./Den));
end

