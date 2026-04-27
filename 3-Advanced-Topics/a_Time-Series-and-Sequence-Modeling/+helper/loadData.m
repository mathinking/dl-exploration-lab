function [X, Y] = loadData(filename, featureIdx)
% loadData  Load battery SOC data from a .mat file.
%   [X, Y] = helper.loadData(FILENAME) loads X and Y from a .mat file
%   and transposes them to (observations x features) format. By default,
%   only the first 3 features (voltage, current, temperature) are returned.
%
%   [X, Y] = helper.loadData(FILENAME, FEATUREIDX) selects features at
%   the specified row indices from the original data.

%   Copyright 2024-2026 The MathWorks, Inc.

arguments
    filename (1,1) string
    featureIdx (1,:) double = 1:3
end

S = load(filename);
X = S.X(featureIdx, :).';   % (observations x numFeatures)
Y = S.Y.';                   % (observations x 1)

end
