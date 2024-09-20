function HNew = UpdatePropContextH(ContextX,ContextW,HCurr,epsilon,Den)
%UPDATECONTEXTH: updates the coefficient matrix HNew = H_{k+1} given the last iterate
%HCurr = H_{k} and the (constant) weight ContextW for the (first K-C) non-discriminative layers 
%of the net. The update rule is that for sparse NMF with beta = 1.

%Num = ContextW'*(ContextX./(max(ContextW*HCurr,epsilon)));  
Num = ContextW'*(ContextX./(ContextW*HCurr+epsilon));  

HNew = max(epsilon, HCurr.*(Num./Den));
end

