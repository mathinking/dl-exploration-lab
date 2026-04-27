function [XSeq, YSeq] = setupDataLSTM(X, Y, subsequenceLength)
% setupDataLSTM  Chunk time-series data into fixed-length subsequences.
%   [XSeq, YSeq] = helper.setupDataLSTM(X, Y, SUBSEQUENCELENGTH) splits
%   X (observations x features) and Y (observations x 1) into cell arrays
%   of subsequences, each of length SUBSEQUENCELENGTH. The last chunk is
%   kept even if shorter than SUBSEQUENCELENGTH.
%
%   XSeq{i} is (timesteps x features) and YSeq{i} is (timesteps x 1),
%   matching MATLAB's convention for sequence data in trainnet.

%   Copyright 2024-2026 The MathWorks, Inc.

arguments
    X (:,:) double
    Y (:,1) double
    subsequenceLength (1,1) double {mustBePositive, mustBeInteger}
end

nObs = size(X, 1);
nChunks = ceil(nObs / subsequenceLength);
XSeq = cell(nChunks, 1);
YSeq = cell(nChunks, 1);

for i = 1:nChunks
    idxStart = (i-1) * subsequenceLength + 1;
    idxEnd = min(i * subsequenceLength, nObs);
    XSeq{i} = X(idxStart:idxEnd, :);   % (timesteps x features)
    YSeq{i} = Y(idxStart:idxEnd);      % (timesteps x 1)
end

end
