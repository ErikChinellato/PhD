function [StateJacobians,DynJacobians] = ComputeJacobians(F,States,Dyn,Inputs,SparseMat,Dynamic,FStateDynInputs,NetParameters)
%COMPUTEJACOBIANS: Computes the jacobians of F at the different layers of
%the net. StateJacobians & DynJacobians are cells of size (1,NetParameters.Layers) where
%StateJacobians{1} = [] since it is not used during backpropagation.

%Variables
Experiment = NetParameters.Experiment;
Layers = NetParameters.Layers;
Jacobians = NetParameters.Jacobians;
N = NetParameters.StateDimension;
P = NetParameters.HiddenDynamicsDimension;

%Setup output
StateJacobians = cell(1,Layers);
DynJacobians = cell(1,Layers);

if strcmp(Jacobians,'Approximated')
    %Approximate jacobians with finite differences

    if strcmp(Experiment,'1')
        for Layer = 2:Layers
            StateJacobians{Layer} = StateJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,NetParameters);
        end
        
        for Layer = 1:Layers
            DynJacobians{Layer} = zeros(N,P(Dynamic));
                                 %DynJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,P(Dynamic),NetParameters);
        end
    end

    if strcmp(Experiment,'2')
        for Layer = 2:Layers
            StateJacobians{Layer} = StateJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,NetParameters);
        end
        
        for Layer = 1:Layers
            DynJacobians{Layer} = zeros(N,P(Dynamic));
                                 %DynJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,P(Dynamic),NetParameters);
        end
    end

    if strcmp(Experiment,'3')
        for Layer = 2:Layers
            StateJacobians{Layer} = StateJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,NetParameters);
        end
        
        for Layer = 1:Layers
            DynJacobians{Layer} = zeros(N,P(Dynamic));
                                 %DynJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,P(Dynamic),NetParameters);
        end
    end

    if strcmp(Experiment,'4')
        for Layer = 2:Layers
            StateJacobians{Layer} = StateJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,NetParameters);
        end
        
        for Layer = 1:Layers
            DynJacobians{Layer} = DynJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,P(Dynamic),NetParameters);
        end
    end

    if strcmp(Experiment,'5')
        for Layer = 2:Layers
            StateJacobians{Layer} = StateJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,NetParameters);
        end
        
        for Layer = 1:Layers
            DynJacobians{Layer} = DynJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,P(Dynamic),NetParameters);
        end
    end

    if strcmp(Experiment,'6')
        for Layer = 2:Layers
            StateJacobians{Layer} = StateJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,NetParameters);
        end
        
        for Layer = 1:Layers
            DynJacobians{Layer} = DynJacobian(F,States{Layer},Dyn,Inputs{Layer},SparseMat,FStateDynInputs{Layer},Layer,N,P(Dynamic),NetParameters);
        end
    end
end

if strcmp(Jacobians,'Algebraic')
    %Set jacobians to their exact algebraic representation, when possible

    if strcmp(Experiment,'1')    
        for Layer = 2:Layers
            %StateJacobians{Layer} = ;
        end

        for Layer = 1:Layers
            %DynJacobians{Layer} = ;
        end
    end

    if strcmp(Experiment,'2')    
        for Layer = 2:Layers
            %StateJacobians{Layer} = ;
        end

        for Layer = 1:Layers
            %DynJacobians{Layer} = ;
        end
    end

    if strcmp(Experiment,'3')    
        for Layer = 2:Layers
            %StateJacobians{Layer} = ;
        end

        for Layer = 1:Layers
            %DynJacobians{Layer} = ;
        end
    end

    if strcmp(Experiment,'4')    
        for Layer = 2:Layers
            %StateJacobians{Layer} = ;
        end

        for Layer = 1:Layers
            %DynJacobians{Layer} = ;
        end
    end

    if strcmp(Experiment,'5')    
        for Layer = 2:Layers
            %StateJacobians{Layer} = ;
        end

        for Layer = 1:Layers
            %DynJacobians{Layer} = ;
        end
    end
end

end

