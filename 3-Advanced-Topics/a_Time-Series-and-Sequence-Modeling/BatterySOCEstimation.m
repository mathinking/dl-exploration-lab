%[text] # Battery State-of-Charge Estimation with Deep Learning
%[text] This script demonstrates how to estimate the State of Charge (SOC) of a lithium-ion battery using deep learning. SOC represents the remaining energy in a battery as a fraction from 0 (empty) to 1 (full) and is a critical quantity in electric vehicles, energy storage systems, and consumer electronics.
%[text] The script covers three progressively more sophisticated approaches:
%[text] 1. A *feedforward neural network* (FFN) baseline that treats each timestep independently, first with raw features, then improved with hand-engineered moving averages.
%[text] 2. A *stacked LSTM network* that learns temporal dynamics from sequences of sensor readings.
%[text] 3. *Monotonic neural networks* that architecturally guarantee physically consistent behavior -- SOC always increases during charging and decreases during discharging. \
%[text] All three approaches use experimental data from LG HG2 18650 battery cells at multiple temperatures.
%[text] Based on the MathWorks examples [Train Network for Battery State of Charge Estimation](https://www.mathworks.com/help/deeplearning/ug/train-network-for-battery-state-of-charge-estimation.html) and [Battery SOC Estimation Using Monotonic Neural Networks](https://github.com/matlab-deep-learning/constrained-deep-learning/blob/main/examples/monotonic/BSOCEstimateUsingMonotonicNetworks/BatteryStateOfChargeEstimationUsingMonotonicNeuralNetworks.md).
%%
%[text] ## Load Data
%[text] Load the LG HG2 battery dataset. The training data contains a single long sequence recorded during four driving cycles at different temperatures (-10°C, 0°C, 10°C, and 25°C). Each sample has five features: voltage (V), current (A), temperature (°C), and moving averages of voltage and current. All features are normalized to \[0, 1\].
%[text] The FFN baseline starts with the three raw features, then adds the moving averages to show the value of feature engineering. The LSTM uses only the three raw features, relying on its recurrent architecture to capture temporal patterns without hand-crafted features.
dataDir = fullfile("data", "LGHG2@n10C_to_25degC");

[XTrain, YTrain] = helper.loadData(fullfile(dataDir, "Train", "TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat"), 1:5);
[XVal, YVal] = helper.loadData(fullfile(dataDir, "Validation", "01_TEST_LGHG2@n10degC_Norm_(05_Inputs).mat"), 1:5);

testFiles = dir(fullfile(dataDir, "Test", "*.mat"));
testTemps = ["-10°C", "0°C", "10°C", "25°C"];
XTest = cell(4,1);
YTest = cell(4,1);
for i = 1:4
    [XTest{i}, YTest{i}] = helper.loadData(fullfile(testFiles(i).folder, testFiles(i).name), 1:5);
end
%%
%[text] ## Explore the Data
%[text] Visualize the training data. The `stackedplot` shows voltage, current, and temperature alongside the target SOC. Notice how SOC ramps down during discharge (driving) and partially recovers during regenerative braking. The multi-temperature training data spans a wide operating range.
nSamples = 30000;
T = array2table([YTrain(1:nSamples) XTrain(1:nSamples,1:3)], ...
    VariableNames=["SOC", "Voltage", "Current", "Temperature"]);

figure
stackedplot(T)
title("Training Data — First " + nSamples + " Samples")
%%
%[text] ## Feedforward Baseline
%[text] Train a feedforward neural network (FFN) as a baseline. Start with the same 3 raw features the LSTM will use later (voltage, current, temperature) to establish a fair comparison point. The FFN treats each timestep independently -- it has no memory of previous timesteps.
%%
%[text] ### Define FFN Architecture
%[text] The network uses a `featureInputLayer` with rescale-symmetric normalization to standardize the input features. A single hidden layer with 30 neurons and ReLU activation provides the nonlinear mapping. The `sigmoidLayer` at the output constrains predictions to \[0, 1\], matching the SOC range.
ffnLayers3 = [
    featureInputLayer(3, Normalization="rescale-symmetric")
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(1)
    sigmoidLayer];
%%
%[text] ### Set FFN Training Options
%[text] The FFN uses a higher learning rate and fewer epochs than the LSTM because it processes individual samples rather than sequences. The large mini-batch size (2^15) accelerates training on the flat feature matrix.
ffnOptions3 = trainingOptions("adam", ...
    InitialLearnRate=0.1, ...
    MaxEpochs=5, ...
    MiniBatchSize=2^15, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal(:,1:3), YVal}, ...
    ValidationFrequency=20, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=false);
%%
%[text] ### Train the FFN
%[text] Train the FFN using mean squared error (MSE) loss. Set `doTrainingFFN` to `true` to train from scratch, or `false` to load a pre-trained model.
doTrainingFFN = true;

if doTrainingFFN %#ok<*UNRCH>
    ffnNet3 = trainnet(XTrain(:,1:3), YTrain, ffnLayers3, "mse", ffnOptions3);
    save(fullfile("models", "ffnBaseline3Features.mat"), "ffnNet3")
else
    load(fullfile("models", "ffnBaseline3Features.mat"), "ffnNet3")
end
%%
%[text] ### Evaluate FFN on Test Data
%[text] Run the trained FFN on a held-out test sequence at the first temperature. Each timestep is predicted independently.
YPredFFN3 = predict(ffnNet3, XTest{1}(:,1:3));

residualFFN3 = YTest{1} - YPredFFN3;
rmseFFN3 = sqrt(mean(residualFFN3.^2));

figure
tiledlayout(2,1)
nexttile
plot(YTest{1}); hold on; plot(YPredFFN3); hold off
legend("True SOC", "FFN Prediction")
title("Feedforward Baseline (3 Features) at " + testTemps(1))
nexttile
plot(residualFFN3)
ylabel("Residual")
title("Residuals — RMSE: " + rmseFFN3)
%%
%[text] ## Improving the FFN with Feature Engineering
%[text] The FFN processes each timestep in isolation, so it has no knowledge of the past. One way to inject temporal context is through *feature engineering*: adding moving averages of voltage and current as extra inputs. These hand-crafted features give the FFN a limited window into recent history.
%[text] Use all 5 features (voltage, current, temperature, plus moving averages of voltage and current) and retrain the FFN.
ffnLayers = [
    featureInputLayer(5, Normalization="rescale-symmetric")
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(1)
    sigmoidLayer];

ffnOptions = trainingOptions("adam", ...
    InitialLearnRate=0.1, ...
    MaxEpochs=5, ...
    MiniBatchSize=2^15, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    ValidationFrequency=20, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=false);

if doTrainingFFN
    ffnNet = trainnet(XTrain, YTrain, ffnLayers, "mse", ffnOptions);
    save(fullfile("models", "feedforwardBaseline.mat"), "ffnNet")
else
    load(fullfile("models", "feedforwardBaseline.mat"), "ffnNet")
end
%%
%[text] ### Compare 3-Feature vs 5-Feature FFN
%[text] Compare the FFN with and without moving average features on the same test sequence.
YPredFFN = predict(ffnNet, XTest{1});

residualFFN = YTest{1} - YPredFFN;
rmseFFN = sqrt(mean(residualFFN.^2));

figure
tiledlayout(2,1)
nexttile
plot(YTest{1}); hold on; plot(YPredFFN3); plot(YPredFFN); hold off
legend("True SOC", "FFN (3 features)", "FFN (5 features)")
title("Effect of Moving Average Features at " + testTemps(1))
nexttile
plot(residualFFN3); hold on; plot(residualFFN); hold off
legend("3 features — RMSE: " + rmseFFN3, "5 features — RMSE: " + rmseFFN)
ylabel("Residual")
title("Residuals")
%%
%[text] The moving averages improve the FFN's accuracy by providing a limited view of the past. However, these features are hand-engineered and only capture a fixed time horizon. This motivates the use of *recurrent networks*, which learn temporal representations directly from the raw sensor data -- no feature engineering required.
%%
%[text] ## Stacked LSTM
%[text] Now train a stacked LSTM that learns temporal dynamics directly from raw sensor data — no feature engineering required. The recurrent architecture gives the network an implicit memory of past timesteps, replacing the hand-crafted moving averages.
%%
%[text] ### Prepare Sequence Data
%[text] LSTMs process sequences, not individual samples. We chunk the long training and validation time-series into fixed-length subsequences. Each chunk becomes one training example, giving the LSTM a window of temporal context for learning charge/discharge dynamics. The `trainnet` function handles left-padding of the last (shorter) chunk automatically.
subsequenceLength = 500;

[XTrainSeq, YTrainSeq] = helper.setupDataLSTM(XTrain(:,1:3), YTrain, subsequenceLength);
[XValSeq, YValSeq] = helper.setupDataLSTM(XVal(:,1:3), YVal, subsequenceLength);
%%
%[text] ### Define LSTM Architecture
%[text] The network uses two stacked LSTM layers with decreasing hidden units (256 → 128), each followed by dropout for regularization. The `sigmoidLayer` at the output constrains predictions to \[0, 1\], matching the normalized SOC range.
%[text] - `sequenceInputLayer(3, Normalization="rescale-symmetric")` — accepts 3-channel time-series input (voltage, current, temperature), rescaling each feature to \[-1, 1\] so the LSTM sees uniform magnitudes regardless of physical units.
%[text] - `lstmLayer(256, OutputMode="sequence")` — first recurrent layer that learns temporal patterns. The 256 hidden units provide capacity for modeling complex dynamics.
%[text] - `dropoutLayer(0.1)` — light dropout (10%) to regularize without underfitting. Higher dropout rates hurt performance with only 3 input features.
%[text] - `lstmLayer(128, OutputMode="sequence")` — second recurrent layer that learns higher-level temporal abstractions from the first layer's output.
%[text] - `fullyConnectedLayer(1)` — maps the 128-dim hidden state to a single SOC prediction at each timestep.
%[text] - `sigmoidLayer` — constrains the output to \[0, 1\]. \
layers = [
    sequenceInputLayer(3, Normalization="rescale-symmetric")
    lstmLayer(256, OutputMode="sequence")
    dropoutLayer(0.1)
    lstmLayer(128, OutputMode="sequence")
    dropoutLayer(0.1)
    fullyConnectedLayer(1)
    sigmoidLayer];
%%
%[text] ### Set LSTM Training Options
%[text] Use Adam optimizer with a piecewise learning rate schedule: start at LR=0.01 for fast initial convergence, then drop 10× every 200 epochs. With 500 epochs, this gives two LR drops (at 200 and 400), spending the majority of training at lower learning rates where the LSTM refines its temporal representations. `GradientThreshold=1` clips gradients to prevent the exploding gradient problem common in RNNs. Left-padding ensures the LSTM processes padding tokens first, then real data — this avoids corrupting the final hidden state with pad values.
options = trainingOptions("adam", ...
    InitialLearnRate=0.01, ...
    MaxEpochs=500, ...
    MiniBatchSize=64, ...
    GradientThreshold=1, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=200, ...
    LearnRateDropFactor=0.1, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    ValidationData={XValSeq, YValSeq}, ...
    ValidationFrequency=50, ...
    Plots="training-progress", ...
    Metrics="rmse", ...
    Verbose=true);
%%
%[text] ### Train the LSTM
%[text] Train the stacked LSTM using mean squared error (MSE) loss. Set `doTraining` to `true` to train from scratch, or `false` to load the pre-trained model.
doTraining = false;

if doTraining %#ok<*UNRCH>
    net = trainnet(XTrainSeq, YTrainSeq, layers, "mse", options);
    save(fullfile("models", "stackedLSTM.mat"), "net")
else
    load(fullfile("models", "stackedLSTM.mat"), "net")
end
%%
%[text] ### Evaluate LSTM on Test Data
%[text] Run the trained LSTM on a held-out test sequence. For inference, we pass the entire test sequence as a single input (no chunking needed). The LSTM processes it timestep-by-timestep, carrying its hidden state across the full sequence.
YPredLSTM = predict(net, XTest{1}(:,1:3));

residualLSTM = YTest{1} - YPredLSTM;
rmseLSTM = sqrt(mean(residualLSTM.^2));

figure
tiledlayout(2,1)
nexttile
plot(YTest{1}); hold on; plot(YPredLSTM); hold off
legend("True SOC", "LSTM Prediction")
title("Stacked LSTM at " + testTemps(1))
nexttile
plot(residualLSTM)
ylabel("Residual")
title("Residuals — RMSE: " + rmseLSTM)
%%
%[text] ## Compare All Models Across Temperatures
%[text] Evaluate all three models on the four test temperatures. The FFN (3 features) uses only raw sensor data, the FFN (5 features) adds hand-engineered moving averages, and the LSTM uses only 3 raw features but captures temporal dynamics through its recurrent architecture. Change `tempIdx` to explore different operating conditions.
rmseResults = zeros(4, 3);
YPredAll = cell(4, 3);
for i = 1:4
    Xi3 = XTest{i}(:,1:3);
    YPredAll{i,1} = predict(ffnNet3, Xi3);
    YPredAll{i,2} = predict(ffnNet, XTest{i});
    YPredAll{i,3} = predict(net, Xi3);
    rmseResults(i,1) = sqrt(mean((YTest{i} - YPredAll{i,1}).^2));
    rmseResults(i,2) = sqrt(mean((YTest{i} - YPredAll{i,2}).^2));
    rmseResults(i,3) = sqrt(mean((YTest{i} - YPredAll{i,3}).^2));
end

tempIdx = 4; % Change to -10°C, 0°C, 10°C, or 25°C %[control:dropdown:0ed5]{"position":[11,12]}

figure
plot(YTest{tempIdx}); hold on
plot(YPredAll{tempIdx,1}); plot(YPredAll{tempIdx,2}); plot(YPredAll{tempIdx,3}); hold off
legend("True", "FFN (3)", "FFN (5)", "LSTM", Location="best")
title("Predictions at " + testTemps(tempIdx))
ylabel("SOC")

figure
plot(YTest{tempIdx} - YPredAll{tempIdx,1}); hold on
plot(YTest{tempIdx} - YPredAll{tempIdx,2})
plot(YTest{tempIdx} - YPredAll{tempIdx,3}); hold off
legend("FFN (3)", "FFN (5)", "LSTM", Location="best")
title("Residuals at " + testTemps(tempIdx))
ylabel("Residual")
%%
%[text] ## Monotonic Networks for Guaranteed Safe Behavior
%[text] In safety-critical applications such as battery management systems, the SOC estimate must obey physical laws: it should *always increase* during charging and *always decrease* during discharging. A standard LSTM does not guarantee this -- it can produce momentary violations where the predicted SOC briefly increases during discharge or vice versa.
%[text] *Monotonic neural networks* enforce this constraint architecturally. The key insight is to predict the *rate of change* of SOC rather than the absolute value:
%[text] - $\\Delta y(t) = y(t+1) - y(t) = g(x(t))$
%[text] - $y(t) = y(1) + \\sum\_{k=1}^{t} g(x(k-1))$ \
%[text] By constraining $g$ to be positive (charging) or negative (discharging), the cumulative sum is guaranteed to be monotonically increasing or decreasing. This constraint holds at every training iteration, including initialization.
%[text] For details, see the [constrained-deep-learning](https://github.com/matlab-deep-learning/constrained-deep-learning) repository.
%%
%[text] ### Split Training Data into Charging/Discharging Phases
%[text] The monotonic constraint requires knowing whether the battery is charging or discharging. The reference approach uses a fixed cycle length, but our data has variable drive cycles. Instead, we detect the overall charge/discharge direction from a smoothed SOC signal (`movmean` with window 1000) and split at the phase boundaries. The smoothing is used *only for direction detection* — the targets remain the raw, unmodified SOC values. Short segments (\< 200 samples) are discarded to match the chunk size.
monoMinLength = 200;
smoothDir = movmean(YTrain, 1000);

[XMonoCharge, XMonoDischarge] = helper.splitByDirection(XTrain(:,1:3), smoothDir, monoMinLength);
[YMonoCharge, YMonoDischarge] = helper.splitByDirection(YTrain, smoothDir, monoMinLength);
%%
%[text] Visualize the charging and discharging phases detected from the smoothed SOC direction.
dY = [0; diff(smoothDir)];
category = sign(dY);
chgMask = category == 1;
dchMask = category == -1;

YChg = YTrain; YChg(~chgMask) = NaN;
YDch = YTrain; YDch(~dchMask) = NaN;

figure
plot(YChg, Color="#0072BD", DisplayName="Charging")
hold on
plot(YDch, Color="#D95319", DisplayName="Discharging")
hold off
legend(Location="best")
title("Training Data — Charging vs Discharging Phases")
xlabel("Sample"); ylabel("SOC")
%%
%[text] ### Preprocess: Chunk, Smooth, and Compute Normalized Diffs
%[text] Chunk each segment into overlapping windows of 200 samples with stride 100 (50% overlap), matching the [reference implementation](https://github.com/matlab-deep-learning/constrained-deep-learning). The charge and discharge paths use different preprocessing: **discharge** chunks are quadratic-smoothed to guarantee monotonic diffs and filtered by direction, while **charge** chunks use raw data. This asymmetry improves cold-temperature accuracy — quadratic smoothing inflates charge diff magnitudes by ~25%, causing cumulative overshoot at temperatures where charging is slower.
chunkSize = 200;
stride = 100;

[XTrainCharge, YTrainCharge] = helper.chunkData(XMonoCharge, YMonoCharge, chunkSize, stride);
[XTrainDischarge, YTrainDischarge] = helper.chunkData(XMonoDischarge, YMonoDischarge, chunkSize, stride);

% Discharge: quadratic smoothing — diffs of a quadratic are linear (monotonic).
% Discard chunks where the fitted quadratic doesn't match the expected direction.
% Charge: use raw data (no smoothing). Quadratic smoothing inflates charge
% diff magnitudes by ~25%, causing cumulative overshoot at cold temperatures.
YTrainDischarge = cellfun(@(y) polyval(polyfit((1:numel(y))', y, 2), (1:numel(y))'), ...
    YTrainDischarge, UniformOutput=false);
isMonoDischarge = cellfun(@(y) all(diff(y) < 0), YTrainDischarge);
XTrainDischarge = XTrainDischarge(isMonoDischarge);
YTrainDischarge = YTrainDischarge(isMonoDischarge);

[XTrainChargeDiff, YTrainChargeDiff, stdDiffCharge] = helper.preprocessDiffTargets(XTrainCharge, YTrainCharge);
[XTrainDischargeDiff, YTrainDischargeDiff, stdDiffDischarge] = helper.preprocessDiffTargets(XTrainDischarge, YTrainDischarge);

% Discard outlier chunks with extreme diffs (sharp cycle-boundary transitions)
% that would dominate the MSE loss and cause training spikes.
maxDiffMag = 5;
isInlierCharge = cellfun(@(y) all(abs(y) < maxDiffMag), YTrainChargeDiff);
XTrainChargeDiff = XTrainChargeDiff(isInlierCharge);
YTrainChargeDiff = YTrainChargeDiff(isInlierCharge);
isInlierDischarge = cellfun(@(y) all(abs(y) < maxDiffMag), YTrainDischargeDiff);
XTrainDischargeDiff = XTrainDischargeDiff(isInlierDischarge);
YTrainDischargeDiff = YTrainDischargeDiff(isInlierDischarge);
%%
%[text] ### Prepare Validation Data for Monotonic Networks
%[text] Apply the same preprocessing pipeline to the validation set so that the validation loss is directly comparable to training loss. Use the training-set standard deviations for diff normalization.
smoothDirVal = movmean(YVal, 1000);
[XValCharge, XValDischarge] = helper.splitByDirection(XVal(:,1:3), smoothDirVal, monoMinLength);
[YValCharge, YValDischarge] = helper.splitByDirection(YVal, smoothDirVal, monoMinLength);

[XValCharge, YValCharge] = helper.chunkData(XValCharge, YValCharge, chunkSize, stride);
[XValDischarge, YValDischarge] = helper.chunkData(XValDischarge, YValDischarge, chunkSize, stride);

YValDischarge = cellfun(@(y) polyval(polyfit((1:numel(y))', y, 2), (1:numel(y))'), ...
    YValDischarge, UniformOutput=false);
XValDischarge = XValDischarge(cellfun(@(y) all(diff(y) < 0), YValDischarge));
YValDischarge = YValDischarge(cellfun(@(y) all(diff(y) < 0), YValDischarge));

XValChargeDiff = cellfun(@(x) x(1:end-1,:), XValCharge, UniformOutput=false);
YValChargeDiff = cellfun(@(y) diff(y) / stdDiffCharge, YValCharge, UniformOutput=false);
XValDischargeDiff = cellfun(@(x) x(1:end-1,:), XValDischarge, UniformOutput=false);
YValDischargeDiff = cellfun(@(y) diff(y) / stdDiffDischarge, YValDischarge, UniformOutput=false);

XValChargeDiff = XValChargeDiff(cellfun(@(y) all(abs(y) < maxDiffMag), YValChargeDiff));
YValChargeDiff = YValChargeDiff(cellfun(@(y) all(abs(y) < maxDiffMag), YValChargeDiff));
XValDischargeDiff = XValDischargeDiff(cellfun(@(y) all(abs(y) < maxDiffMag), YValDischargeDiff));
YValDischargeDiff = YValDischargeDiff(cellfun(@(y) all(abs(y) < maxDiffMag), YValDischargeDiff));
%%
%[text] ### Monotonic Network Architecture
%[text] Two separate LSTM networks are trained: one for charging and one for discharging. Both use two stacked LSTM layers (128 and 64 hidden units) with dropout, but differ in their output constraint and input normalization.
%[text] **Input normalization.** The charging network uses `zscore` normalization, while the discharging network uses `rescale-zero-one`. The training data is heavily imbalanced by temperature — approximately 96% of charging segments come from ~25°C. With `rescale-zero-one`, features are scaled to \[0, 1\] based on the training range, so test data at 10°C maps to near-zero values that the network has rarely seen. With `zscore`, features are centered and scaled by their standard deviation, which preserves the relative magnitude of deviations from the training mean and produces better generalization to underrepresented temperatures. The discharging data is more balanced across temperatures, so `rescale-zero-one` works well there.
%[text] **Output constraints:**
%[text] - *Charging*: a `reluLayer` ensures $\\Delta y \> 0$ (SOC increases during charging).
%[text] - *Discharging*: the pattern `scalingLayer(-1)` → `reluLayer` → `scalingLayer(-1)` computes $-\\text{ReLU}(-h(x))$, ensuring $\\Delta y \< 0$ (SOC decreases during discharging). \
numFeatures = size(XTrainChargeDiff{1}, 2);

chargingLayers = [
    sequenceInputLayer(numFeatures, Normalization="zscore")
    lstmLayer(128)
    dropoutLayer(0.2)
    lstmLayer(64)
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    reluLayer];

dischargingLayers = [
    sequenceInputLayer(numFeatures, Normalization="rescale-zero-one")
    lstmLayer(128)
    dropoutLayer(0.2)
    lstmLayer(64)
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    scalingLayer(Scale=-1)
    reluLayer
    scalingLayer(Scale=-1)];
%%
%[text] ### Train Monotonic Networks
%[text] Train both networks using Adam with a warm-up + cosine learning rate schedule.
%[text] **Loss functions.** The charging network is trained with **MAE (mean absolute error)** loss, while the discharging network uses **MSE (mean squared error)**. Because the monotonic architecture predicts per-timestep differences ($\\Delta y$) that are cumulatively summed to reconstruct SOC, small systematic biases in the predictions compound over thousands of timesteps. With the temperature-imbalanced charging data, MSE's quadratic penalty causes the network to over-fit to the dominant 25°C regime, producing a ~9% systematic underprediction of $\\Delta y$ at 10°C that accumulates into large SOC errors. MAE's linear penalty treats all error magnitudes equally, which reduces this temperature-dependent drift. The discharging data is more balanced across temperatures, so MSE works well and provides stronger gradient signal for convergence.
chargeEpochs = 100;
dischargeEpochs = 50;
warmupEpochs = 5;
chargeLR = {warmupLearnRate(NumSteps=warmupEpochs, FrequencyUnit="epoch"), ...
    cosineLearnRate(Period=chargeEpochs - warmupEpochs, PeriodGrowthFactor=Inf, FrequencyUnit="epoch")};
dischargeLR = {warmupLearnRate(NumSteps=warmupEpochs, FrequencyUnit="epoch"), ...
    cosineLearnRate(Period=dischargeEpochs - warmupEpochs, PeriodGrowthFactor=Inf, FrequencyUnit="epoch")};

monoOptionsCharge = trainingOptions("adam", ...
    MaxEpochs=chargeEpochs, ...
    GradientThreshold=1, ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule=chargeLR, ...
    MiniBatchSize=32, ...
    GradientDecayFactor=0.90, ...
    ValidationData={XValChargeDiff, YValChargeDiff}, ...
    ValidationFrequency=5, ...
    Verbose=true, ...
    Plots="training-progress", ...
    Shuffle="every-epoch");

monoOptionsDisch = trainingOptions("adam", ...
    MaxEpochs=dischargeEpochs, ...
    GradientThreshold=1, ...
    InitialLearnRate=0.001, ...
    LearnRateSchedule=dischargeLR, ...
    MiniBatchSize=32, ...
    GradientDecayFactor=0.95, ...
    ValidationData={XValDischargeDiff, YValDischargeDiff}, ...
    ValidationFrequency=5, ...
    Verbose=true, ...
    Plots="training-progress", ...
    Shuffle="every-epoch");
doTrainingMono = false;

if doTrainingMono %#ok<*UNRCH>
    chargingNet = trainnet(XTrainChargeDiff, YTrainChargeDiff, chargingLayers, "mae", monoOptionsCharge);
    dischargingNet = trainnet(XTrainDischargeDiff, YTrainDischargeDiff, dischargingLayers, "mse", monoOptionsDisch);
    save(fullfile("models", "chargingConstrainedNet.mat"), "chargingNet", "stdDiffCharge")
    save(fullfile("models", "dischargingConstrainedNet.mat"), "dischargingNet", "stdDiffDischarge")
else
    load(fullfile("models", "chargingConstrainedNet.mat"), "chargingNet", "stdDiffCharge")
    load(fullfile("models", "dischargingConstrainedNet.mat"), "dischargingNet", "stdDiffDischarge")
end
%%
%[text] ### Evaluate Monotonic Networks
%[text] Split the test data using a smoothed SOC direction (matching the training split approach). Each segment is independently anchored at the true SOC at its start, so prediction errors do not compound across segments. This mirrors a real BMS where SOC is periodically recalibrated (e.g., at full charge or after rest periods).
XTestMono = XTest{tempIdx}(:,1:3);
YTestMono = YTest{tempIdx};
smoothTestDir = movmean(YTestMono, 1000);
[XTestCharge, XTestDischarge, ~, ~, chPosTest, dchPosTest] = ...
    helper.splitByDirection(XTestMono, smoothTestDir, monoMinLength);
[YTestCharge, YTestDischarge] = ...
    helper.splitByDirection(YTestMono, smoothTestDir, monoMinLength);

YPredMono = nan(numel(YTestMono), 1);

for i = 1:numel(XTestCharge)
    xSeq = dlarray(XTestCharge{i}(1:end-1,:)', 'CT');
    Yout = double(extractdata(predict(chargingNet, xSeq)))';
    yRecon = [YTestCharge{i}(1); YTestCharge{i}(1) + cumsum(stdDiffCharge * Yout)];
    YPredMono(chPosTest(i,1):chPosTest(i,2)) = yRecon;
end

for i = 1:numel(XTestDischarge)
    xSeq = dlarray(XTestDischarge{i}(1:end-1,:)', 'CT');
    Yout = double(extractdata(predict(dischargingNet, xSeq)))';
    yRecon = [YTestDischarge{i}(1); YTestDischarge{i}(1) + cumsum(stdDiffDischarge * Yout)];
    YPredMono(dchPosTest(i,1):dchPosTest(i,2)) = yRecon;
end

validMono = ~isnan(YPredMono);
rmseMono = sqrt(mean((YTestMono(validMono) - YPredMono(validMono)).^2));

% LSTM prediction on the same test data for comparison
YPredLSTMMono = predict(net, XTestMono);
rmseLSTMMono = sqrt(mean((YTestMono - YPredLSTMMono).^2));

figure
plot(YTestMono); hold on; plot(YPredLSTMMono); plot(YPredMono); hold off
legend("True SOC", "LSTM (RMSE: " + rmseLSTMMono + ")", "Monotonic (RMSE: " + rmseMono + ")", Location="best")
title("LSTM vs Constrained (Monotonic) Network at " + testTemps(tempIdx))
xlabel("Sample"); ylabel("SOC")
%%
%[text] ### Monotonicity Score
%[text] The *monotonicity score* measures how well a prediction obeys the monotonic constraint within each charge/discharge phase. A score of 1.0 means every consecutive pair follows the expected direction (increasing during charge, decreasing during discharge). The constrained network should achieve near-perfect scores since it is architecturally constrained, while unconstrained models may produce violations. Phases are defined by the raw true SOC direction, with short runs (\< 20 samples) filtered out.
modelNames = ["FFN (3)", "FFN (5)", "LSTM", "Monotonic"];
YPredTemp = {predict(ffnNet3, XTestMono), ...
             predict(ffnNet, XTest{tempIdx}), ...
             YPredLSTMMono, ...
             YPredMono};
fprintf("Monotonicity scores at %s:\n", testTemps(tempIdx))
for m = 1:4
    score = helper.monotonicityScore(YPredTemp{m}, smoothTestDir, monoMinLength);
    fprintf("  %-12s %.4f\n", modelNames(m), score)
end
%%
%[text] ## Model Comparison — RMSE by Temperature
%[text] Evaluate all four models (FFN 3-feature, FFN 5-feature, LSTM, and monotonic constrained network) across the four test temperatures. The monotonic network uses per-segment anchoring so that prediction errors do not compound across charge/discharge phases.
rmseAll = zeros(4, 4);
monoScoreAll = zeros(4, 4);
YPredAllFinal = cell(4, 4);
for i = 1:4
    Xi3 = XTest{i}(:,1:3);
    YPredAllFinal{i,1} = predict(ffnNet3, Xi3);
    YPredAllFinal{i,2} = predict(ffnNet, XTest{i});
    YPredAllFinal{i,3} = predict(net, Xi3);

    % Monotonic prediction — per-segment anchoring
    smoothDirI = movmean(YTest{i}, 1000);
    [XTCh, XTDch, ~, ~, chPosI, dchPosI] = helper.splitByDirection(Xi3, smoothDirI, monoMinLength);
    [YTCh, YTDch] = helper.splitByDirection(YTest{i}, smoothDirI, monoMinLength);

    YPredMonoI = nan(numel(YTest{i}), 1);

    for j = 1:numel(XTCh)
        xSeq = dlarray(XTCh{j}(1:end-1,:)', 'CT');
        Yout = double(extractdata(predict(chargingNet, xSeq)))';
        yRecon = [YTCh{j}(1); YTCh{j}(1) + cumsum(stdDiffCharge * Yout)];
        YPredMonoI(chPosI(j,1):chPosI(j,2)) = yRecon;
    end

    for j = 1:numel(XTDch)
        xSeq = dlarray(XTDch{j}(1:end-1,:)', 'CT');
        Yout = double(extractdata(predict(dischargingNet, xSeq)))';
        yRecon = [YTDch{j}(1); YTDch{j}(1) + cumsum(stdDiffDischarge * Yout)];
        YPredMonoI(dchPosI(j,1):dchPosI(j,2)) = yRecon;
    end

    YPredAllFinal{i,4} = YPredMonoI;

    % RMSE: for monotonic, compute only over predicted (non-NaN) regions
    for m = 1:3
        rmseAll(i,m) = sqrt(mean((YTest{i} - YPredAllFinal{i,m}).^2));
    end
    validI = ~isnan(YPredMonoI);
    rmseAll(i,4) = sqrt(mean((YTest{i}(validI) - YPredMonoI(validI)).^2));

    % Monotonicity scores (using smoothed SOC direction for fair comparison)
    for m = 1:4
        monoScoreAll(i,m) = helper.monotonicityScore(YPredAllFinal{i,m}, smoothDirI, monoMinLength);
    end
end

resultsAllTable = table(testTemps', rmseAll(:,1), rmseAll(:,2), rmseAll(:,3), rmseAll(:,4), ...
    VariableNames=["Temperature", "FFN_3feat", "FFN_5feat", "LSTM", "Monotonic_LSTM"])
monoScoreTable = table(testTemps', monoScoreAll(:,1), monoScoreAll(:,2), monoScoreAll(:,3), monoScoreAll(:,4), ...
    VariableNames=["Temperature", "FFN_3feat", "FFN_5feat", "LSTM", "Monotonic_LSTM"])
figure
tiledlayout(1,2)
nexttile
bar(rmseAll)
set(gca, XTickLabel=testTemps)
xlabel("Temperature"); ylabel("RMSE")
title("RMSE by Temperature")
nexttile
bar(monoScoreAll)
set(gca, XTickLabel=testTemps)
xlabel("Temperature"); ylabel("Monotonicity Score")
ylim([0 1])
title("Monotonicity Score by Temperature")
lgd = legend("FFN (3)", "FFN (5)", "LSTM", "Monotonic LSTM");
lgd.Layout.Tile = "south";
%%
%[text] ## Export Results
%[text] Uncomment the lines below to export this script as an HTML report.
%if ~isfolder("results"), mkdir("results"); end
%export("BatterySOCEstimation.m", fullfile("results", "BatterySOCEstimation.html"));
%%
%[text] *Copyright 2024-2026 The MathWorks, Inc.*

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
%[control:dropdown:0ed5]
%   data: {"defaultValue":"1","itemLabels":["-10ºC","0ºC","10ºC","25ºC"],"items":["1","2","3","4"],"label":"tempIdx","run":"Section"}
%---
