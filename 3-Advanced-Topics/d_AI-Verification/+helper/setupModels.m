% setupModels — Prepare models for the AI Verification exercise
% Downloads the ACAS Xu neural networks and converts to MAT format.

rootDir = fileparts(fileparts(mfilename('fullpath')));

%% Verify required toolboxes
requiredToolboxes = "Deep Learning Toolbox";
installedToolboxes = string({ver().Name});
missingToolboxes = setdiff(requiredToolboxes, installedToolboxes);
if ~isempty(missingToolboxes)
    warning("Missing toolboxes: %s", join(missingToolboxes, ", "));
else
    disp("All required toolboxes are installed.");
end

%% Verify required add-ons
requiredAddons = "AI Verification Library for Deep Learning Toolbox";
addons = matlab.addons.installedAddons;
missingAddons = setdiff(requiredAddons, addons.Name);
if ~isempty(missingAddons)
    warning("Missing add-ons: %s" + newline + ...
        "  Install via Add-On Explorer.", join(missingAddons, ", "));
else
    disp("All required add-ons are installed.");
end

%% Download ACAS Xu networks if needed
modelsDir = fullfile(rootDir, "models");
if isempty(dir(fullfile(modelsDir, "*.mat")))
    disp("Downloading ACAS Xu neural networks...");

    zipFile = matlab.internal.examples.downloadSupportFile( ...
        "nnet", "data/acas-xu-neural-network-dataset.zip");

    disp("Extracting networks...");
    unzip(zipFile, modelsDir);

    % Move contents out of the zip subfolder into models/
    zipSubfolder = fullfile(modelsDir, "acas-xu-neural-network-dataset");
    movefile(fullfile(zipSubfolder, "*"), modelsDir);
    rmdir(zipSubfolder);

    disp("Converting ONNX networks to MAT format...");
    onnxFolder = fullfile(modelsDir, "networks-onnx");
    helper.convertACASXuFromONNXAndSave(onnxFolder, modelsDir);

    disp("ACAS Xu neural network download and conversion complete.");
else
    disp("ACAS Xu neural networks already present.");
end

disp("Setup complete.");
