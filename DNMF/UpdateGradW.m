function [PosGradW,NegGradW] = UpdateGradW(PosGradHCompB,NegGradHCompB,PosGradWCompA,NegGradWCompA,GradWCompC,PosGradWCompD,NegGradWCompD,H)
%UPDATEGRADW: updates the positive and negative parts of the gradient for W.

PosGradW = PosGradWCompA + PosGradHCompB*H' + GradWCompC + PosGradWCompD;
NegGradW = NegGradWCompA + NegGradHCompB*H' + GradWCompC + NegGradWCompD;
end

