classdef CubeBatchQueue < handle
%CubeBatchQueue FIFO queue of CubeBatch objects for adaptive mesh
%   Manages a queue of CubeBatch objects, allowing batched retrieval
%   up to a specified size for efficient GPU processing.
%
%   Example:
%     q = helper.CubeBatchQueue;
%     enqueue(q, cubes);
%     batch = getCubes(q, 1024);

    properties (Access = private)
        batches = {}
    end

    methods
        function enqueue(obj, cubes)
        %enqueue Add a CubeBatch to the end of the queue
            obj.batches{end+1} = cubes;
        end

        function cubes = getCubes(obj, maxSize)
        %getCubes Retrieve up to maxSize hypercuboids from the queue
            collected = helper.CubeBatch();
            remaining = maxSize;

            while ~isempty(obj.batches) && remaining > 0
                batch = obj.batches{1};
                obj.batches(1) = [];
                n = getNumCubes(batch);

                if n <= remaining
                    collected = addSubset(collected, batch);
                    remaining = remaining - n;
                else
                    collected = addSubset(collected, getSubset(batch, 1:remaining));
                    leftover = getSubset(batch, remaining+1:n);
                    obj.batches = [{leftover} obj.batches];
                    remaining = 0;
                end
            end

            cubes = collected;
        end

        function tf = isempty(obj)
        %isempty Check if the queue has no remaining hypercuboids
            tf = builtin('isempty', obj.batches);
        end
    end
end
