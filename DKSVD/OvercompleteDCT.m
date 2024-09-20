function Dict= OvercompleteDCT(n,m)
%OVERCOMPLETEDCT: computes the overcomplete discrete cosine transform
%dictionary of size (n^2,m^2).

Dict = zeros(n, m);

for ColInd = 1:m
    Col = cos( (0:n-1)*( (ColInd-1)*pi)/m );
    if ColInd > 1
        Col = Col - mean(Col);
    end

    Dict(:,ColInd) = Col'/norm(Col,2);
end

Dict = kron(Dict, Dict);
Dict = normalize(Dict,'norm',2);

indx = reshape(reshape(1:n^2,n,n)',1,n^2);
Dict = Dict(indx, :);
end

