function net = loadACASNetwork(previousAdvisory, tau)
%loadACASNetwork Load an ACAS Xu neural network by advisory and tau index
%   net = loadACASNetwork(previousAdvisory, tau) loads the ACAS Xu network
%   for the given previous advisory (1-5) and time-until-loss-of-vertical-
%   separation index (1-9).
%
%   Previous advisory mapping:
%     1 = Clear-of-Conflict, 2 = Weak Left, 3 = Weak Right,
%     4 = Strong Left, 5 = Strong Right
%
%   Tau ranges from 1 (imminent collision) to 9 (distant threat).

rootDir = fileparts(fileparts(mfilename('fullpath')));
matFolder = fullfile(rootDir, "models");
filename = sprintf("ACASXU_run2a_%d_%d_batch_2000.mat", ...
    previousAdvisory, tau);
filepath = fullfile(matFolder, filename);

if ~isfile(filepath)
    error("helper:networkNotFound", ...
        "Network file not found: %s\nRun helper.setupModels to download the networks.", ...
        filepath);
end

s = load(filepath);
net = s.net;
end
