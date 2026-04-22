function [net, validationLoss] = train(net, mbqTrain, mbqValidation, opts)
arguments
    net
    mbqTrain
    mbqValidation
    opts.NumIterations = 1000;
    opts.LearnRate = 1e-3;
    opts.MinLearnRate = 0;
    opts.WarmupIterations = 0;
    opts.MaxGradientNorm = Inf;
    opts.WeightDecay = 0;
    opts.Beta1 = 0.9;
    opts.Beta2 = 0.999;
    opts.ValidationFrequency = 100;
    opts.NumValidationIters = 20;
    opts.ValidationPatience = Inf;
    opts.Verbose = false;
end

% Determine if LR schedule is active
useLRSchedule = opts.WarmupIterations > 0;

averageGrad = [];
averageSqGrad = [];

% Precompute weight decay mask (skip biases and layer norm parameters)
if opts.WeightDecay > 0
    learnables = net.Learnables;
    decayMask = true(size(learnables, 1), 1);
    for ii = 1:size(learnables, 1)
        paramName = learnables.Parameter{ii};
        if contains(paramName, "Bias") || ...
                strcmp(paramName, "Scale") || ...
                strcmp(paramName, "Offset")
            decayMask(ii) = false;
        end
    end
end

bestValLoss = Inf;
patienceCounter = 0;

stopRequested = false;
monitor = trainingProgressMonitor( ...
    Metrics=["TrainingLoss", "ValidationLoss"], ...
    XLabel="Iteration");
groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);

if opts.Verbose
    if useLRSchedule
        fprintf("  Progress   | Elapsed  | Training Loss | Validation Loss | Learn Rate\n");
        fprintf("-------------|----------|---------------|-----------------|----------\n");
    else
        fprintf("  Progress   | Elapsed  | Training Loss | Validation Loss\n");
        fprintf("-------------|----------|---------------|----------------\n");
    end
    tStart = tic;
end

iteration = 0;
while iteration < opts.NumIterations && ~stopRequested
    iteration = iteration + 1;

    % Compute learning rate for this iteration
    if useLRSchedule
        learnRate = computeLearnRate(iteration, opts.NumIterations, ...
            opts.LearnRate, opts.MinLearnRate, opts.WarmupIterations);
    else
        learnRate = opts.LearnRate;
    end

    [X,T] = mbqTrain.next();
    [loss, gradients] = dlfeval(@nanogpt.modelLoss, net, X, T);

    % Gradient clipping by global norm
    if isfinite(opts.MaxGradientNorm)
        gradients = clipGradients(gradients, opts.MaxGradientNorm);
    end

    [net,averageGrad,averageSqGrad] = adamupdate(net,gradients, ...
            averageGrad,averageSqGrad,iteration,learnRate,opts.Beta1,opts.Beta2);

    % Decoupled weight decay (AdamW)
    if opts.WeightDecay > 0
        learnableValues = net.Learnables.Value;
        scale = 1 - learnRate * opts.WeightDecay;
        for ii = find(decayMask)'
            learnableValues{ii} = learnableValues{ii} * scale;
        end
        net.Learnables.Value = learnableValues;
    end

    recordMetrics(monitor,iteration,TrainingLoss=double(extractdata(loss)));
    stopRequested = monitor.Stop;

    % Every validation frequency, validate on numValidationIters minibatches from the validation set
    if mod(iteration, opts.ValidationFrequency) == 0 || iteration == 1
        validationLoss = 0;
        for ii = 1:opts.NumValidationIters
            [XVal, TVal] = mbqValidation.next();
            validationLoss = validationLoss + nanogpt.modelLoss(net, XVal, TVal)/opts.NumValidationIters;
        end

        % Early stopping check
        valLossScalar = double(extractdata(validationLoss));
        if valLossScalar < bestValLoss
            bestValLoss = valLossScalar;
            patienceCounter = 0;
        else
            patienceCounter = patienceCounter + 1;
        end

        recordMetrics(monitor,iteration,ValidationLoss=valLossScalar);
        if opts.Verbose
            elapsed = duration(0,0,toc(tStart),Format="hh:mm:ss");
            if useLRSchedule
                fprintf(" %5d/%-5d | %s | %13.4f | %15.4f | %.2e\n", ...
                    iteration, opts.NumIterations, elapsed, double(extractdata(loss)), valLossScalar, learnRate);
            else
                fprintf(" %5d/%-5d | %s | %13.4f | %15.4f\n", ...
                    iteration, opts.NumIterations, elapsed, double(extractdata(loss)), valLossScalar);
            end
        end

        if patienceCounter >= opts.ValidationPatience
            if opts.Verbose
                fprintf("Early stopping: validation loss did not improve for %d checks.\n", opts.ValidationPatience);
            end
            break;
        end
    end
end
end

function lr = computeLearnRate(iteration, numIterations, maxLR, minLR, warmupIterations)
% Linear warmup followed by cosine decay
if iteration <= warmupIterations
    % Linear warmup from 0 to maxLR
    lr = maxLR * (iteration / warmupIterations);
else
    % Cosine decay from maxLR to minLR
    decayIterations = numIterations - warmupIterations;
    progress = (iteration - warmupIterations) / decayIterations;
    lr = minLR + 0.5 * (maxLR - minLR) * (1 + cos(pi * progress));
end
end

function gradients = clipGradients(gradients, maxNorm)
% Clip gradients by global norm
globalNorm = 0;
for ii = 1:numel(gradients.Value)
    globalNorm = globalNorm + sum(gradients.Value{ii}(:).^2);
end
globalNorm = sqrt(extractdata(globalNorm));

if globalNorm > maxNorm
    scale = maxNorm / globalNorm;
    for ii = 1:numel(gradients.Value)
        gradients.Value{ii} = gradients.Value{ii} * scale;
    end
end
end