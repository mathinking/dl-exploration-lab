classdef CubeBatch
%CubeBatch Batch of hypercuboids for adaptive mesh verification
%   A CubeBatch stores lower and upper bounds for a collection of
%   hypercuboids as dlarray matrices of size numFeatures-by-numCubes.
%
%   Example:
%     cubes = helper.CubeBatch(dlarray(xL,"BC"), dlarray(xU,"BC"));
%     centers = getCenters(cubes);
%     n = getNumCubes(cubes);

    properties
        xLower
        xUpper
    end

    methods
        function obj = CubeBatch(xLower, xUpper)
            if nargin == 0
                obj.xLower = [];
                obj.xUpper = [];
            else
                obj.xLower = xLower;
                obj.xUpper = xUpper;
            end
        end

        function centers = getCenters(obj)
        %getCenters Compute the center point of each hypercuboid
            centers = (obj.xLower + obj.xUpper) / 2;
        end

        function subset = getSubset(obj, idx)
        %getSubset Extract a subset of hypercuboids by index
            subset = helper.CubeBatch(obj.xLower(:, idx), obj.xUpper(:, idx));
        end

        function obj = addSubset(obj, newCubes)
        %addSubset Append hypercuboids from another CubeBatch
            if getNumCubes(newCubes) == 0
                return
            end
            if getNumCubes(obj) == 0
                obj = newCubes;
            else
                obj.xLower = [obj.xLower newCubes.xLower];
                obj.xUpper = [obj.xUpper newCubes.xUpper];
            end
        end

        function n = getNumCubes(obj)
        %getNumCubes Return the number of hypercuboids in this batch
            if isempty(obj.xLower)
                n = 0;
            else
                n = size(obj.xLower, 2);
            end
        end
    end
end
