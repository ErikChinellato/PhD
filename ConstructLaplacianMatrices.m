function [L,LtL] = ConstructLaplacianMatrices(N,SamplingTimes)
%CONSTRUCTLAPLACIANMATRICES: Constructs discrete Laplacian matrices for loss
%function term.

%Variables
TimeStep = SamplingTimes(1); %We are assuming constant sampling

InvTSSq = (1/TimeStep)^2;

%Assemble the matrices
L = [ [2,-5,4,-1,zeros(1,N-4)]; [ [1;zeros(N-3,1)], diag( -2*ones(1,N-2)) + diag(ones(1,N-3),1) + diag(ones(1,N-3),-1), [zeros(N-3,1);1] ]; [zeros(1,N-4),-1,4,-5,2] ];
%L = InvTSSq*L;
LtL = L'*L;
end

