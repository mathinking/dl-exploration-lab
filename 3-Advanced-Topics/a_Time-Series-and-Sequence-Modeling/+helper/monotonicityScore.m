function score = monotonicityScore(YPred, direction, minLength)
% monotonicityScore  Fraction of consecutive pairs obeying monotonic direction.
%   SCORE = helper.monotonicityScore(YPRED, DIRECTION, MINLENGTH) splits
%   YPRED into charging and discharging segments using DIRECTION, then
%   computes the fraction of consecutive pairs where the prediction
%   increases during charging and decreases during discharging.
%
%   A score of 1.0 means perfect monotonicity within every segment.

%   Copyright 2024-2026 The MathWorks, Inc.

arguments
    YPred (:,1) double
    direction (:,1) double
    minLength (1,1) double = 200
end

[chgData, dchData] = helper.splitByDirection(YPred, direction, minLength);

chgDiffs = cellfun(@(x) diff(x), chgData, UniformOutput=false);
dchDiffs = cellfun(@(x) diff(x), dchData, UniformOutput=false);

nCorrect = sum(cellfun(@(x) sum(x >= 0), chgDiffs)) ...
         + sum(cellfun(@(x) sum(x <= 0), dchDiffs));
nTotal = sum(cellfun(@numel, chgDiffs)) ...
       + sum(cellfun(@numel, dchDiffs));

if nTotal == 0
    score = NaN;
else
    score = nCorrect / nTotal;
end

end
