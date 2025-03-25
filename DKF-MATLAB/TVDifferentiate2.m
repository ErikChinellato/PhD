function [uEst,fEst] = TVDifferentiate2(h,alpha,B,D,AtA,Atf,uEst,maxit)
%TVDIFFERENTIATE: Computes u = f' using TV normalization. f must be a
%column vector.

%Detect number of nodes
epsilon = 1e-8;

%Start iterating
for it = 1:maxit
    L = alpha*h*D'*diag( 1./( sqrt( ( (1/2)*D*uEst ).^2 + epsilon ) ) )*D;
    H = L + AtA;                            %Hessian
    g = -( AtA*uEst - Atf + L*uEst );       %-Gradient

    uEst = uEst + ( H\g );
end

%Return denoised f
if nargout > 1
    fEst = B*uEst;
end

end

