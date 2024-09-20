function [CumulativePeriodogram,Freqs] = TestBartlett(Signal)
%
% test di Bartlett (o del periodogramma cumulato): serve per verificare se una sequenza è rumore bianco
%
% Una sequenza x(k) generata da rumore bianco presenta una sequenza di autocorrelazione in cui tutti i
% campioni per shift > 0 sono nulli. Quindi, osservare valori della sequenza di auto-correlazione
% significativamente (in senso statistico) non-nulli corrisponde alla presenza di componenti
% non-random nella sequenza x(k). Per individuare componenti periodiche mischiate al rumore è 
% vantaggioso considerare il "periodogramma cumulato":
%

N = length(Signal);

%% Compute PSD with Welch's method
T = 1;
SeqNum = 1;

NSeq = N/SeqNum;
% periodogramma modificato:
Signal = reshape(Signal,NSeq,SeqNum);
Win = 0.54 - 0.46 * cos( 2*pi*(0:NSeq-1)/(NSeq-1)); % finestra di Hamming
WinNorm = norm(Win)^2/NSeq;  % valore quadratico medio della sequenza di finestratura dei dati
Signal = Signal.*repmat(Win', 1, SeqNum);
PSD = sum( abs(fft(Signal)).^2,2 )/( SeqNum*NSeq*WinNorm );
Freq = 0:( 1/(NSeq*T) ):( 1/(2*T) - 1/(NSeq*T) );

%% Compute normalized cumulative PSD
EstMean = mean(Signal);
EstVar = sum((Signal - EstMean).^2)/(N-1);
CumulativePeriodogram = cumsum( PSD(2:length(Freq)) )/( N*EstVar );
CumulativePeriodogram = [0; CumulativePeriodogram/CumulativePeriodogram(end)]'; %Normalization

if nargout > 1
    Freqs = Freq;
end

end
