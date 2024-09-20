function [PosGradHCompA,PosGradHCompB,NegGradHCompA,NegGradHCompB,PosGradWCompA,NegGradWCompA,GradWCompC,PosGradWCompD,NegGradWCompD] = ReoccurringComponents(X,W,H,PosGradH,NegGradH,epsilon,SparsePen,m,N)
%REOCCURRINGCOMPONENTS: computes some reoccurring components in the H and W
%gradient update formulas for efficiency.

Lam = W*H;
XdLam = X./(Lam+epsilon);
%XdLamSq = XdLam./(Lam+epsilon);
XdLamSq = X./(Lam.^2+epsilon);
WtXdLam = W'*XdLam;

Den = repmat(sum(W)',1,N) + SparsePen; %This was already computed in the PropagateInput function... maybe save them for efficiency? 
WtXdLamdDen = WtXdLam./Den;
HdDen = H./Den;

PosGradHCompA = WtXdLamdDen.*PosGradH;
NegGradHCompA = WtXdLamdDen.*NegGradH;
PosGradHCompB = XdLamSq.*( W*( HdDen.*NegGradH ) );
NegGradHCompB = XdLamSq.*( W*( HdDen.*PosGradH ) );

PosGradWCompA = XdLam*(HdDen.*PosGradH)';
NegGradWCompA = XdLam*(HdDen.*NegGradH)';
GradWCompC = 0*-W.*( XdLamSq*( H.*HdDen.*( PosGradH+NegGradH ) )' );
PosGradWCompD = repmat(sum(HdDen.*NegGradHCompA,2)',m,1);
NegGradWCompD = repmat(sum(HdDen.*PosGradHCompA,2)',m,1);
end