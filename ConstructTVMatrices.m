function [A,D,AtA,B] = ConstructTVMatrices(N,SamplingTimes)
%CONSTRUCTTVMATRICES: Construct matrices used during TV regularization.

%Variables
TimeStep = SamplingTimes(1); %We are assuming constant sampling

%Set up matrices
vD = zeros(N-1,1);
vD(end) = 1;
MD = diag(-ones(1,N-1)) + diag(ones(1,N-2),1);
D = (2/TimeStep)*[ MD, vD ];

vA = ones(N-1,1);
vA(1) = 3/4;
CoeffVecA = [1/4,7/4,2*ones(1,N-3)];
MA = zeros(N-1,N-1);
for DiagInd = 0:N-2
    MA = MA + diag(CoeffVecA(DiagInd+1)*ones(1,N-1-DiagInd),-DiagInd);
end
A = (TimeStep/2)*[ vA, MA ];
AtA = A'*A;

vB = ones(N-1,1);
CoeffVecB = [1,2*ones(1,N-2)];
MB = zeros(N-1,N-1);
for DiagInd = 0:N-2
    MB = MB + diag(CoeffVecB(DiagInd+1)*ones(1,N-1-DiagInd),-DiagInd);
end
B = (TimeStep/2)*[ vB, MB ];
end

