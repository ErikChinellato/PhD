function F = DynamicEquation(z,x,p,u,s,M,K,d,Ts,NetParameters)
%DYNAMICEQUATION: Encodes the (implicit) equation for the dynamics to be solved
%forward in time.

%Variables
Experiment = NetParameters.Experiment;

%Dictionary for model discovery
Phi = ConstructDictionary(z,NetParameters);

if strcmp(Experiment,'1')
    F = M*( z - x ) - Ts*( K*z + d + s'*Phi' );
end

if strcmp(Experiment,'2')
    F = M*( z - x ) - Ts*( K*z + d + s'*Phi' );
end

if strcmp(Experiment,'3')
    F = M*( z - x ) - Ts*( K*z + [0;-z(1)*z(3);z(1)*z(2)] + d + s'*Phi' );
end

if strcmp(Experiment,'4')
    M(1,1) = M(1,1) + p;
    F = M*( z - x ) - Ts*( K*z + [0;-z(1)*z(3);z(1)*z(2)] + d + s'*Phi' );
end

if strcmp(Experiment,'5')
    K(2,1) = K(2,1) + p;
    F = M*( z - x ) - Ts*( K*z + [0;-z(1)*z(3);z(1)*z(2)] + d + s'*Phi' );
end

if strcmp(Experiment,'6')
    F = M*( z - x ) - Ts*( K*z + d + s'*Phi' );
end

end

