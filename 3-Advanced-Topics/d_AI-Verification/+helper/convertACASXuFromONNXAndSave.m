function convertACASXuFromONNXAndSave(onnxFolder, outputFolder)
%convertACASXuFromONNXAndSave Convert ACAS Xu ONNX networks to MAT format
%   convertACASXuFromONNXAndSave(onnxFolder, outputFolder) imports the
%   learnable parameters from each ONNX network in onnxFolder, manually
%   constructs an initialized dlnetwork, and saves to outputFolder.

oldDir = cd(outputFolder);
cleanup = onCleanup(@() cd(oldDir));

onnxFiles = dir(fullfile(onnxFolder, "*.onnx"));

for ii = 1:numel(onnxFiles)
    [~, name] = fileparts(onnxFiles(ii).name);
    matPath = fullfile(outputFolder, name + ".mat");
    if ~isfile(matPath)
        onnxPath = fullfile(onnxFolder, onnxFiles(ii).name);
        params = iImportParametersFromONNX(onnxPath);
        net = iConstructACASXuNetwork(params);
        save(matPath, "net");
    end

    if mod(ii, 10) == 0
        fprintf("    .");
    else
        fprintf(".");
    end
end
fprintf("\n");
end

%% Local functions
function params = iImportParametersFromONNX(onnxFile)
evalc('params = importONNXFunction(onnxFile, "tmp_params.m")');
delete("tmp_params.m");
end

function net = iConstructACASXuNetwork(params)
fc1 = fullyConnectedLayer(50, ...
    Weights=extractdata(params.Learnables.W0)', ...
    Bias=extractdata(params.Nonlearnables.B0), Name="fc_1");
fc2 = fullyConnectedLayer(50, ...
    Weights=extractdata(params.Learnables.W1)', ...
    Bias=extractdata(params.Nonlearnables.B1), Name="fc_2");
fc3 = fullyConnectedLayer(50, ...
    Weights=extractdata(params.Learnables.W2)', ...
    Bias=extractdata(params.Nonlearnables.B2), Name="fc_3");
fc4 = fullyConnectedLayer(50, ...
    Weights=extractdata(params.Learnables.W3)', ...
    Bias=extractdata(params.Nonlearnables.B3), Name="fc_4");
fc5 = fullyConnectedLayer(50, ...
    Weights=extractdata(params.Learnables.W4)', ...
    Bias=extractdata(params.Nonlearnables.B4), Name="fc_5");
fc6 = fullyConnectedLayer(50, ...
    Weights=extractdata(params.Learnables.W5)', ...
    Bias=extractdata(params.Nonlearnables.B5), Name="fc_6");
fc7 = fullyConnectedLayer(5, ...
    Weights=-extractdata(params.Learnables.W6)', ...
    Bias=-extractdata(params.Nonlearnables.B6), Name="fc_7");

layers = [
    featureInputLayer(5, Name="input")
    fc1
    reluLayer(Name="relu_1")
    fc2
    reluLayer(Name="relu_2")
    fc3
    reluLayer(Name="relu_3")
    fc4
    reluLayer(Name="relu_4")
    fc5
    reluLayer(Name="relu_5")
    fc6
    reluLayer(Name="relu_6")
    fc7
    ];

net = dlnetwork(layers);
end
