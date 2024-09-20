function y = dSigmoid(x)
%DSIGMOID: Computes the (elementwise) derivative of sigmoid applied to x.

y = exp(-x)/( (1 + exp(-x))^2 );
end

