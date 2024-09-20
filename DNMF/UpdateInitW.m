function WNew = UpdateInitW(X,W,H,epsilon)
%UPDATEINITW: updates the dictionary for a given source in the
%initialization procedure of the net's weights. The update rule is that for 
%NMF with beta = 1 or 2.

%Beta = 2
%WNew = max( epsilon, W.*( ( X*H' )./( W*(H*H') + epsilon ) ) );

%Beta = 1
Num = (X./(W*H+epsilon))*H';  
Den = repmat(sum(H,2)',size(W,1),1); 

WNew = max( epsilon, W.*( Num./Den ) );
end

