function [XChunked, YChunked] = chunkData(XCell, YCell, chunkSize, stride)
% chunkData  Split cell arrays of sequences into overlapping chunks.
%   [XCHUNKED, YCHUNKED] = helper.chunkData(XCELL, YCELL, CHUNKSIZE,
%   STRIDE) splits each sequence in the cell arrays into overlapping chunks
%   of length CHUNKSIZE with step size STRIDE. Chunks that would be shorter
%   than CHUNKSIZE are discarded.

%   Based on the constrained-deep-learning example by The MathWorks, Inc.
%   Copyright 2024-2026 The MathWorks, Inc.

XChunked = [];
YChunked = [];

for i = 1:numel(XCell)
    X = XCell{i};
    Y = YCell{i};

    numSamples = length(Y);
    numChunks = floor(numSamples / (chunkSize - stride)) - 1;
    XC = cell(numChunks, 1);
    YC = cell(numChunks, 1);

    for j = 1:numChunks
        idxStart = 1 + (j-1) * stride;
        idxEnd = idxStart + chunkSize - 1;
        XC{j} = X(idxStart:idxEnd, :);
        YC{j} = Y(idxStart:idxEnd);
    end

    XChunked = [XChunked; XC]; %#ok<AGROW>
    YChunked = [YChunked; YC]; %#ok<AGROW>
end

end
