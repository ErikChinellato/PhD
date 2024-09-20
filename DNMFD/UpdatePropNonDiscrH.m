function HNew = UpdatePropNonDiscrH(XdXHat,W,HCurr,Den,Epsilon)
%UPDATEPROPNONDISCRH: updates the coefficient matrix HNew = H_{Source,Layer+1} given the last iterate
%HCurr = H_{Source,Layer} and the (constant) weight W = W_{Source} for the (first K-C) non-discriminative layers 
%of the net. The update rule is that for sparse NMFD with beta = 1.

Num = Convolve(W,XdXHat);

HNew = max( Epsilon, HCurr.*( Num/Den ) );
end

