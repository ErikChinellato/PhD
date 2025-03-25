function StateJac = StateJacobian(F,x,p,u,s,Fxpu,Layer,N,NetParameters)
%STATEJACOBIAN: Computes the jacobian matrix for F with respect to the x 
%variables at the point (x,p,u). We are assuming F to be vector valued with
%the same dimension as the input x. Fxpu = F(x,p,u) is given as an input for
%efficiency since it was already computed (NOT USED IF NetParameters.FiniteDifferences = 'Central').

%Variables
FiniteDifferences = NetParameters.FiniteDifferences;
h = NetParameters.FiniteDifferencesSkip;

StateJac = zeros(N);

%Cycle over columns of jacobian
for ColInd = 1:N
    %Increment in ColInd-th cardinal direction
    Increment = zeros(N,1);
    Increment(ColInd) = h;

    if strcmp(FiniteDifferences,'Forward') == 1
        StateJac(:,ColInd) = (F(x+Increment,p,u,s,Layer,NetParameters) - Fxpu )/h;
    end

    if strcmp(FiniteDifferences,'Backward') == 1
        StateJac(:,ColInd) = ( Fxpu - F(x-Increment,p,u,s,Layer,NetParameters) )/h;
    end

    if strcmp(FiniteDifferences,'Central') == 1
        StateJac(:,ColInd) = ( F(x+Increment,p,u,s,Layer,NetParameters) - F(x-Increment,p,u,s,Layer,NetParameters) )/( 2*h );
    end
end

end

