function bisectAndUpdateQueue(queue, cubes, minAngle)
%bisectAndUpdateQueue Bisect unproven hypercuboids and add to queue
%   bisectAndUpdateQueue(queue, cubes, minAngle) bisects each hypercuboid
%   in cubes along its widest angular dimension (theta or psi) and
%   enqueues the resulting sub-cuboids. Hypercuboids narrower than
%   minAngle in both angular dimensions are discarded.

n = getNumCubes(cubes);
if n == 0
    return
end

xL = extractdata(cubes.xLower);
xU = extractdata(cubes.xUpper);

newLower = [];
newUpper = [];

for ii = 1:n
    widths = xU(:, ii) - xL(:, ii);
    % Only bisect angular dimensions (2 = theta, 3 = psi)
    [maxW, bestIdx] = max(widths([2 3]));
    bestDim = bestIdx + 1;

    if maxW < minAngle
        continue
    end

    mid = (xL(bestDim, ii) + xU(bestDim, ii)) / 2;

    % First half
    lo1 = xL(:, ii); hi1 = xU(:, ii);
    hi1(bestDim) = mid;

    % Second half
    lo2 = xL(:, ii); hi2 = xU(:, ii);
    lo2(bestDim) = mid;

    newLower = [newLower lo1 lo2]; %#ok<AGROW>
    newUpper = [newUpper hi1 hi2]; %#ok<AGROW>
end

if ~isempty(newLower)
    newCubes = helper.CubeBatch( ...
        dlarray(newLower, "CB"), dlarray(newUpper, "CB"));
    enqueue(queue, newCubes);
end
end
