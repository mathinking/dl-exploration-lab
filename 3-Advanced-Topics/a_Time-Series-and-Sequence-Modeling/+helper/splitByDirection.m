function [chargeData, dischargeData, chargeIdx, dischargeIdx, chargePos, dischargePos] = splitByDirection(data, direction, minLength)
% splitByDirection  Split time-series into charging and discharging segments.
%   [CHARGE, DISCHARGE, CIDX, DIDX] = helper.splitByDirection(DATA,
%   DIRECTION, MINLENGTH) splits DATA into cell arrays of contiguous
%   charging (SOC increasing) and discharging (SOC decreasing) segments,
%   based on the strict sign of diff(DIRECTION). Regions where
%   diff(DIRECTION) == 0 (rest periods) are skipped.
%
%   [~, ~, ~, ~, CPOS, DPOS] = helper.splitByDirection(...) also returns
%   the [start, end] row positions of each kept segment in the original
%   DATA array. Use these to reassemble predictions at the correct
%   positions when some segments are dropped by the MINLENGTH filter.
%
%   DATA      — array to split (obs x cols). Can be features, targets, or
%               predictions.
%   DIRECTION — SOC signal (obs x 1) whose derivative determines the
%               charge/discharge direction. When splitting features or
%               predictions, pass the true SOC as DIRECTION.
%   MINLENGTH — minimum segment length to keep (default: 50). Short
%               segments (e.g., noise or brief regenerative braking) are
%               discarded.

%   Copyright 2024-2026 The MathWorks, Inc.

arguments
    data (:,:) double
    direction (:,1) double
    minLength (1,1) double = 50
end

dY = [0; diff(direction)];
category = sign(dY);  % +1 = charging, -1 = discharging, 0 = rest

% Find boundaries where category changes
changes = find(diff(category) ~= 0);
boundaries = [1; changes + 1; size(data, 1) + 1];

chargeData = {};
dischargeData = {};
chargeIdx = [];
dischargeIdx = [];
chargePos = zeros(0, 2);
dischargePos = zeros(0, 2);
segCount = 0;

for k = 1:numel(boundaries) - 1
    s = boundaries(k);
    e = boundaries(k + 1) - 1;
    if (e - s + 1) < minLength
        continue
    end
    segCount = segCount + 1;
    if category(s) == 1
        chargeData{end + 1, 1} = data(s:e, :); %#ok<AGROW>
        chargeIdx(end + 1, 1) = segCount; %#ok<AGROW>
        chargePos(end + 1, :) = [s, e]; %#ok<AGROW>
    elseif category(s) == -1
        dischargeData{end + 1, 1} = data(s:e, :); %#ok<AGROW>
        dischargeIdx(end + 1, 1) = segCount; %#ok<AGROW>
        dischargePos(end + 1, :) = [s, e]; %#ok<AGROW>
    end
    % category == 0 (rest): skip
end

end
