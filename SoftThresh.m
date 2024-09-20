function y = SoftThresh(Theta,Vec)
%SOFTTRESH: Computes the (elementwise) soft-threshold of Vec with parameter Theta.

y = sign(Vec).*max( abs(Vec) - Theta, 0 );
end

