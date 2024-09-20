function XHat = ConstructXHat(Weights,H)
%CONSTRUCTXHAT: Contructs XHat given the dictionaries Weights and
%coefficients H.

XHat = zeros(size(Weights{1},1),size(H{1},2));

for Source = 1:length(Weights)
    XHat = XHat + Convolve(Weights{Source},H{Source});
end

end

