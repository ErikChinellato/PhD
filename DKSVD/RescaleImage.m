function Y = RescaleImage(X)
%RESCALEIMAGE: Rescales a grayscale image in [-1,1].

Mean = 255/2;
Std = Mean;

Y = (X-Mean)/Std;
end

