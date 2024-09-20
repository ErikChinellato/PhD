function [Lambda,Mu,Nu,Csi] = MLP(y,b0A0,b1A1,b2A2,b3A3,NetParameters)
%MLP: Computes lambda, mu, nu and csi for the given weights and patch y (vectorized row-wise).

%Variables
MLPLastActivation = NetParameters.MLPLastActivation;

%Propagate y
Nu = max( b0A0*[1;y], 0 );
Mu = max( b1A1*[1;Nu], 0 );
Csi = max( b2A2*[1;Mu], 0 );

if strcmp(MLPLastActivation,'Identity')
    Lambda = b3A3*[1;Csi];
end

if strcmp(MLPLastActivation,'ReLU')
    Lambda = max( b3A3*[1;Csi], 0 );
end

if strcmp(MLPLastActivation,'Sigmoid')
    Lambda = 1/( 1 + exp( -b3A3*[1;Csi] ) );
end

end

