function [plotAxis] = plotSpacecraftAxes(img, points_camera_plane)
% plotSpacecraftAxes Overlay spacecraft coordinate axes on an image.
%   plotAxis = plotSpacecraftAxes(img, points_camera_plane) draws the x-axis
%   (red), y-axis (green), and z-axis (blue) arrows on the image and returns
%   the annotated image without requiring figure rendering.
%
%   Author(s): Reece Teramoto, Kautilya Vemulapalli
%   Copyright 2024 The MathWorks, Inc.

x = points_camera_plane(1:4);
y = points_camera_plane(5:8);

origin = [x(1) y(1)];
plotAxis = img;

% Draw axes as lines: red = x, green = y, blue = z
plotAxis = insertShape(plotAxis, "line", [origin x(2) y(2)], Color="red",   LineWidth=4);
plotAxis = insertShape(plotAxis, "line", [origin x(3) y(3)], Color="green", LineWidth=4);
plotAxis = insertShape(plotAxis, "line", [origin x(4) y(4)], Color="blue",  LineWidth=4);
end
