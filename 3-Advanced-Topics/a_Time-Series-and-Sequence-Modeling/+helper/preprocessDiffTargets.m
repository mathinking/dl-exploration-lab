function [XDiff, YDiff, stdDiff] = preprocessDiffTargets(X, Y)
% preprocessDiffTargets  Preprocess targets as normalized differences.
%   [XDIFF, YDIFF, STDDIFF] = helper.preprocessDiffTargets(X, Y) computes
%   the difference between consecutive SOC values, normalizes by the
%   standard deviation (no offset, preserving sign), and removes the final
%   input value to match dimensions.
%
%   Use STDDIFF to apply inverse normalization during inference.

%   Based on the constrained-deep-learning example by The MathWorks, Inc.
%   Copyright 2024-2026 The MathWorks, Inc.

YDiff = cellfun(@(x) diff(x), Y, UniformOutput=false);

stdDiff = std(cell2mat(YDiff));
YDiff = cellfun(@(x) x / stdDiff, YDiff, UniformOutput=false);

% Remove the final value in each input sequence to match dimensions
XDiff = cellfun(@(x) x(1:end-1, :), X, UniformOutput=false);

end
