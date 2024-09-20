function GradH = UpdateGradH(GradHCompA,GradHCompB,NetParameters)
%UPDATEGRADH: Updates the gradients with respects to the current layer's H
%matrix ('s rows).

%Variables
S = NetParameters.Sources;

GradH = cell(2,S);

for Source = 1:S
    for Sign = 1:2
        GradH{Sign,Source} = GradHCompA{Sign,Source} + GradHCompB{Sign,Source};
    end
end

end

