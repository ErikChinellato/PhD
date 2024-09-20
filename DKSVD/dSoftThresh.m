function [dThetaST, dVecST] = dSoftThresh(Theta,Vec)
%DSOFTTHRESH: Computes the (elementwise) derivative of soft-threshold of Vec 
%with parameter Theta.

dThetaST = ones(size(Vec));
dVecST = ones(size(Vec));

if Theta >= 0
    dThetaST( (Vec >= -Theta)&(Vec <= Theta) ) = 0;
    dThetaST( Vec > Theta  ) = -1;
    dVecST( (Vec >= -Theta)&(Vec <= Theta) ) = 0;  
else
    dThetaST( Vec >= 0 ) = -1;
end

%dThetaST = -sign(SoftThresh(Theta,Vec));

end

