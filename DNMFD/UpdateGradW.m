function Grads = UpdateGradW(GradWCompA,GradWCompB,GradWCompC,Grads,NetParameters)
%UPDATEGRADW: Updates the gradients with respects to the current layer's 
%weights and adds them to the minibatch.

%Variables
S = NetParameters.Sources;

%Update the gradients and add them to the minibatch
for Source = 1:S
    for Sign = 1:2
        Grads{Sign,Source} = Grads{Sign,Source} + GradWCompA{Sign,Source} + GradWCompB{Sign,Source} + GradWCompC{Sign,Source};
    end
end

end