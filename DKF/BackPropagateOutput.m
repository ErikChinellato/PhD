function Grads = BackPropagateOutput(StateTrue,Dynamic,States,MeasurementMinusCStates,GainMeasurementMinusCFs,MeasurementMinusCFs,TensorizedGains,MeasurementWeightMatsSym,PredictorWeightMatsSym,Grads,StateJacobians,DynJacobians,NetWeights,NetParameters)
%BACKPROPAGATEOUTPUT: Computes the gradients of the loss function with
%respects to the parameters. The loss function is:
%
%       (Penalty0/2)*||States{Layers+1}-StateTrue||^2 + 
%                          + sum_{Layer = 1,...,Layers}(Penalty1/2)*( MeasurementMinusCStates{Layer}' )*MeasurementWeightMats{Layer}*( MeasurementMinusCStates{Layer} ) + 
%                          + sum_{Layer = 1,...,Layers}(Penalty2/2)*( GainMeasurementMinusCFs{Layer}' )*PredictorWeightMats{Layer}*( GainMeasurementMinusCFs{Layer} ) +
%                          + (Penalty3/2)*||L*TensorizedGains||^2 = 
%       = Eps + sum_{Layer = 1,...,Layers} F^Layer + sum_{Layer = 1,...,Layers} G^Layer + H
%
%We can choose to backpropagate the gradients stemming from F^Layer, G^Layer or to
%truncate them at the Layer-th layer only.

%Variables
Layers = NetParameters.Layers;
C = NetParameters.C; 
LtL = NetParameters.LtL;
StateDimension = NetParameters.StateDimension;
SharedWeights = NetParameters.SharedWeights;
BackPropagation = NetParameters.BackPropagation;
Penalty0 = NetParameters.Penalty0;
Penalty1 = NetParameters.Penalty1;
Penalty2 = NetParameters.Penalty2;
Penalty3 = NetParameters.Penalty3;

GradsStateEps = cell(1);
GradsStateF = cell(1,Layers);
GradsStateG = cell(1,Layers);

%Compute the gradients with respects to the net's weights and update the 
%gradients with respects to the states along the way
for Layer = Layers:-1:1
    if strcmp(SharedWeights,'Yes')
        Indx = 1;
    end
    if strcmp(SharedWeights,'No')
        Indx = Layer;
    end

    %Construct common matrix components
    CommonMat = -NetWeights{Indx}*C;
    if Layer > 1
        CommonMatState = CommonMat*StateJacobians{Layer};
    end
    CommonMatDyn = CommonMat*DynJacobians{Layer};
    CommonMat = eye(StateDimension) + CommonMat;

    %Construct the dynamics gradient matrix
    DynMat = ( CommonMat*DynJacobians{Layer} )';

    %Construct the gradient update matrix at current layer
    if Layer > 1
        UpdateMat = ( CommonMat*StateJacobians{Layer} )';
    end

    %Gradient of H with respect to NetWeights{Indx}
    Grads{Indx} = Grads{Indx} + Penalty3*tensorprod(TensorizedGains,LtL(Layer,:),3,2);
    
    if strcmp(BackPropagation,'Complete')

        if Layer == Layers
            %Gradient of Eps with respect to state at last layer
            GradsStateEps{1} = Penalty0*( States{end} - StateTrue );
        end
        %Gradient of Eps with respect to NetWeights{Indx}
        Grads{Indx} = Grads{Indx} + GradsStateEps{1}*MeasurementMinusCFs{Layer}';
        %Gradient of Eps with respect to NetWeights{end}{Dynamic}
        Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateEps{1};

        %Gradient of F^Layer with respect to state at Layer-th layer
        GradsStateF{Layer} = -Penalty1(Layer)*( C'*MeasurementWeightMatsSym{Layer}*MeasurementMinusCStates{Layer} );
        %Gradient of F^Layer with respect to NetWeights{Indx}
        Grads{Indx} = Grads{Indx} + GradsStateF{Layer}*MeasurementMinusCFs{Layer}';
        %Gradient of F^Layer with respect to NetWeights{end}{Dynamic}
        Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateF{Layer};

        %Gradient of G^Layer with respect to fictitious state at Layer-th layer
        GradsStateG{Layer} = Penalty2(Layer)*PredictorWeightMatsSym{Layer}*GainMeasurementMinusCFs{Layer};
        %Gradient of G^Layer with respect to NetWeights{Indx}
        Grads{Indx} = Grads{Indx} + GradsStateG{Layer}*MeasurementMinusCFs{Layer}';
        %Gradient of G^Layer with respect to NetWeights{end}{Dynamic}
        Grads{end}{Dynamic} = Grads{end}{Dynamic} + CommonMatDyn'*GradsStateG{Layer};
   
        if Layer > 1
            %Update gradient of Eps with respect to state
            GradsStateEps{1} = UpdateMat*GradsStateEps{1};
            %Update gradient of F^Layer with respect to state
            GradsStateF{Layer} = UpdateMat*GradsStateF{Layer};
            %Update gradient of G^Layer with respect to state
            GradsStateG{Layer} = CommonMatState'*GradsStateG{Layer};
        end

        for PastLayer = Layer+1:Layers
            %Gradient of F^PastLayer with respect to NetWeights{Indx}
            Grads{Indx} = Grads{Indx} + GradsStateF{PastLayer}*MeasurementMinusCFs{Layer}';
            %Gradient of F^PastLayer with respect to NetWeights{end}{Dynamic}
            Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateF{PastLayer};

            %Gradient of G^PastLayer with respect to NetWeights{Indx}
            Grads{Indx} = Grads{Indx} + GradsStateG{PastLayer}*MeasurementMinusCFs{Layer}';
            %Gradient of G^PastLayer with respect to NetWeights{end}{Dynamic}
            Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateG{PastLayer};
            
            if Layer > 1
                %Update gradient of F^PastLayer with respect to state
                GradsStateF{PastLayer} = UpdateMat*GradsStateF{PastLayer};
                %Update gradient of G^PastLayer with respect to state
                GradsStateG{PastLayer} = UpdateMat*GradsStateG{PastLayer};
            end
        end
    end

    if strcmp(BackPropagation,'Truncated')
        %Adjust with gradient of Epsilon, F^Layer, G^Layer with respect to NetWeights{Indx}
        %and truncate its backpropagation.

        if Layer == Layers
            %Gradient of Eps with respect to state at last layer
            GradsStateEps{1} = Penalty0*( States{end} - StateTrue );

            %Gradient of Eps with respect to NetWeights{Indx}
            Grads{Indx} = Grads{Indx} + GradsStateEps{1}*MeasurementMinusCFs{end}';
        end
        %Gradient of Eps with respect to NetWeights{end}{Dynamic}
        Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateEps{1};

        %Gradient of F^Layer with respect to state at Layer-th layer
        GradsStateF{Layer} = -Penalty1(Layer)*( C'*MeasurementWeightMatsSym{Layer}*MeasurementMinusCStates{Layer} );
        %Gradient of F^Layer with respect to NetWeights{Indx}
        Grads{Indx} = Grads{Indx} + GradsStateF{Layer}*MeasurementMinusCFs{Layer}';
        %Gradient of F^Layer with respect to NetWeights{end}{Dynamic}
        Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateF{Layer};

        %Gradient of G^Layer with respect to fictitious state at Layer-th layer
        GradsStateG{Layer} = Penalty2(Layer)*PredictorWeightMatsSym{Layer}*GainMeasurementMinusCFs{Layer};
        %Gradient of G^Layer with respect to NetWeights{Indx}
        Grads{Indx} = Grads{Indx} + GradsStateG{Layer}*MeasurementMinusCFs{Layer}';
        %Gradient of G^Layer with respect to NetWeights{end}{Dynamic}
        Grads{end}{Dynamic} = Grads{end}{Dynamic} + CommonMatDyn'*GradsStateG{Layer};
   
        if Layer > 1
            %Update gradient of Eps with respect to state
            GradsStateEps{1} = UpdateMat*GradsStateEps{1};
            %Update gradient of F^Layer with respect to state
            GradsStateF{Layer} = UpdateMat*GradsStateF{Layer};
            %Update gradient of G^Layer with respect to state
            GradsStateG{Layer} = CommonMatState'*GradsStateG{Layer};
        end

        for PastLayer = Layer+1:Layers
            %Gradient of F^PastLayer with respect to NetWeights{end}{Dynamic}
            Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateF{PastLayer};

            %Gradient of G^PastLayer with respect to NetWeights{end}{Dynamic}
            Grads{end}{Dynamic} = Grads{end}{Dynamic} + DynMat*GradsStateG{PastLayer};
            
            if Layer > 1
                %Update gradient of F^PastLayer with respect to state
                GradsStateF{PastLayer} = UpdateMat*GradsStateF{PastLayer};
                %Update gradient of G^PastLayer with respect to state
                GradsStateG{PastLayer} = UpdateMat*GradsStateG{PastLayer};
            end
        end
    end
end
end

