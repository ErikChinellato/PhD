function HNew = UpdateInitH(X,W,H,SparsePen,epsilon)
%UPDATEINITH: updates the coefficients' matrix for a given source in the
%initialization procedure of the net's weights. The update rule is that for 
%sparse NMF with beta = 1 or 2.

%Beta = 2
%HNew = max( epsilon, H.*( max( epsilon, W'*X - SparsePen )./( (W'*W)*H + epsilon ) ) ); %Convergence guarantee
%HNew = max( epsilon, H.*( ( W'*X )./( W'*W*H + SparsePen ) ) ); %Heuristic

%Beta = 1
Num = W'*(X./(W*H+epsilon));  
Den = repmat(sum(W)',1,size(H,2)) + SparsePen; 

HNew = max( epsilon, H.*( Num./Den ) );
end
