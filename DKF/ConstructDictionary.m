function Phi = ConstructDictionary(X,NetParameters)
%CONSTRUCTDICTIONARY: Constructs dictionary for the model discovery.
global matlab_have_dictionary

%Variables
AllowedDictionaryBlocks = NetParameters.AllowedDictionaryBlocks;
DictionaryBlocks = NetParameters.DictionaryBlocks;

[StateDimension,TimeInd] = size(X);
Phi = [];

if any(strcmp(DictionaryBlocks,'Constant'))
    ConstantBlock = ones(TimeInd,1);

    Phi = [Phi,ConstantBlock];
end

if any(strcmp(DictionaryBlocks,'Linear'))
    LinearBlock = X';
    
    Phi = [Phi,LinearBlock];
end

if any(strcmp(DictionaryBlocks,'Quadratic'))
    Counter = 1;
    if matlab_have_dictionary
        QuadraticBlock = zeros(TimeInd,AllowedDictionaryBlocks('Quadratic'));
    else
        QuadraticBlock = zeros(TimeInd,AllowedDictionaryBlocks.Quadratic);
    end
    for i = 1:StateDimension
        for j = i:StateDimension
            QuadraticBlock(:,Counter) = X(i,:)'.*X(j,:)';
            Counter = Counter + 1;
            %QuadraticBlock(:,(i-1)*StateDimension - (i-2)*(i-1)/2 + (j-i+1)) = X(i,:)'.*X(j,:)';
        end
    end

    Phi = [Phi,QuadraticBlock];
end

if any(strcmp(DictionaryBlocks,'Cubic'))
    Counter = 1;
    if matlab_have_dictionary
        CubicBlock = zeros(TimeInd,AllowedDictionaryBlocks('Cubic'));
    else
        CubicBlock = zeros(TimeInd,AllowedDictionaryBlocks.Cubic);
    end
    for i = 1:StateDimension
        for j = i:StateDimension
            for k = j:StateDimension
                CubicBlock(:,Counter) = X(i,:)'.*X(j,:)'.*X(k,:)';
                Counter = Counter + 1;
                %CubicBlock(:, ( (i-1)*(StateDimension+1)*StateDimension - (2*StateDimension+1)*(i-2)*(i-1)/2 + (i-2)*(i-1)*(2*i-3)/6 )/2 + (j-i)*(StateDimension-i+1) - (j-i-1)*(j-i)/2 + (k-j+1) ) = X(i,:)'.*X(j,:)'.*X(k,:)';
            end
        end
    end

    Phi = [Phi,CubicBlock];
end

end

