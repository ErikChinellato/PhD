function HNew = UpdatePropDiscrH(XdXHat,W,HCurr,SparsePen,Epsilon)
%UPDATEH: updates the coefficient matrix HNew = H_{Source,Layer+1} given the last iterate
%HCurr = H_{Source,Layer} and the weight W_{Source,Layer} for the (last C) discriminative layers 
%of the net. The update rule is that for sparse NMFD with beta = 1.

Num = Convolve(W,XdXHat);
Den = sum( W,'all' ) + SparsePen;

HNew = max( Epsilon, HCurr.*( Num/Den ) );
end

