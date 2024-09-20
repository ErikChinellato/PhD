function xp = F(x,p,u,s,Layer,NetParameters)
%F: Computes the state prediction xp.

%Variables
Model = NetParameters.Model;

%Equation for the dynamics
DynEq = @(z)DynamicEquation(z,x,p,u,s,Model.M,Model.K,Model.D(:,Layer),Model.SamplingTimes(Layer),NetParameters);

%Solve for the state prediction
options = optimoptions('fsolve','FunctionTolerance',1e-32,'OptimalityTolerance',1e-16,'StepTolerance',1e-10,'MaxFunctionEvaluations',1e4,'Display','off');
xp = fsolve(DynEq,x,options);

end

