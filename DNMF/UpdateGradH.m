function [PosGradH,NegGradH] = UpdateGradH(PosGradHCompA,PosGradHCompB,NegGradHCompA,NegGradHCompB,W)
%UPDATEGRADH: updates the positive and negative parts of the gradient for H.

PosGradH = PosGradHCompA + W'*PosGradHCompB;
NegGradH = NegGradHCompA + W'*NegGradHCompB;
end

