%[text] # Build a Generative Pretrained Transformer (GPT) in MATLAB
%[text] This demo shows how to build a transformer model to predict the next character in a sequence, based on the works of Shakespeare or Miguel de Cervantes (you choose). The demo is based on the online lecture [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy.
%%
%[text] ## Data Exploration
%[text] Select a text corpus to train on. Two options are available:
%[text] - [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) -- a compilation of Shakespeare's plays (~1.1 MB, 65 unique characters), curated by Andrej Karpathy.
%[text] - [Don Quijote](https://babel.upm.es/~angel/teaching/pps/quijote.txt) -- Miguel de Cervantes' novel in the original Spanish (~2.1 MB, 114 unique characters). \
corpusFileName = "tinyshakespeare"; %[control:dropdown:903b]{"position":[18,35]}
corpusFileNamePath = fullfile("data", corpusFileName + ".txt");
%[text] Import the data as a character array and tokenize it. The tokenizer maps each unique character to an integer (e.g., 'a'$\\rightarrow$1, 'b'$\\rightarrow$2, ...). This is the simplest possible tokenizer -- one token per character. Production models like GPT-2 and GPT-3 use subword tokenizers (Byte Pair Encoding) with vocabularies of ~50,000 tokens, but character-level tokenization keeps things simple for learning.
rawData = fileread(corpusFileNamePath);
if corpusFileName == "quijote"
    rawData(1:27256) = [];
end

tokenizer = nanogpt.CharacterTokenizer(rawData);
data = tokenizer.char2tok(rawData);
%%
%[text] Partition the data by holding out the last 10% of the data as a validation set. The validation set lets us detect *overfitting* -- when the model memorizes the training data rather than learning generalizable patterns. If training loss keeps dropping but validation loss starts rising, the model is overfitting.
rng(1)
validationFraction = 0.1;
splitIdx = floor(numel(data)*(1-validationFraction));
dataTrain = data(1:splitIdx);
dataValidation = data(splitIdx+1:end);
%[text] Create training and validation datastores with a subsequence length of eight. A single chunk of length eight actually packs *eight training examples* into it. For a chunk `[18 47 56 57 1 15 47 58]`, the model learns: given `[18]` predict `47`; given `[18 47]` predict `56`; given `[18 47 56]` predict `57`; and so on up to the full context. This teaches the model to make predictions from contexts of length 1 through `subsequenceLength`. The targets are always the input sequence shifted by one position -- the classic next-token prediction setup.
subsequenceLength = 8;
dsTrain = nanogpt.SequenceForecastingDatastore(dataTrain, subsequenceLength, tokenizer.VocabSize);
dsValidation = nanogpt.SequenceForecastingDatastore(dataValidation, subsequenceLength, tokenizer.VocabSize);
%[text] Preview the first few elements of the training set as characters.
tbl = dsTrain.preview();
tokenizer.tok2char(tbl.predictors)
tokenizer.tok2char(tbl.responses)
%[text] Create minibatchqueues for the training and validation data. Minibatches group multiple independent sequences together so they can be processed in parallel on the GPU. The examples within a batch don't interact -- they're just a way to use hardware more efficiently.
miniBatchSize = 128;
mbqTrain = minibatchqueue(dsTrain, ...
    MinibatchSize=miniBatchSize, ...
    MiniBatchFcn=@(X,T) preprocessMinibatch(X,T), ...
    MiniBatchFormat={'BTC', 'BTC'});
mbqValidation = minibatchqueue(dsValidation, ...
    MinibatchSize=miniBatchSize, ...
    MiniBatchFcn=@(X,T) preprocessMinibatch(X,T), ...
    MiniBatchFormat={'BTC', 'BTC'});
%%
%[text] ## Train Bigram Model
%[text] As a baseline, train a bigram model, which guesses the next letter based only on the one immediately before it. In a bigram model, tokens don't "talk" to each other at all -- each token independently looks up a probability distribution over the next token. This is the simplest possible language model.
%[text] The network is just a single embedding table of size `vocabSize` $\\times$ `vocabSize`. Each row stores the learned probability distribution for what comes after that character. For example, row 5 might learn that after the letter 'e', a space is most likely.
bigramNet = dlnetwork( ...
    [ ...
    sequenceInputLayer(1), ...
    wordEmbeddingLayer(tokenizer.VocabSize, tokenizer.VocabSize), ...
    softmaxLayer()]);
%[text] Before training, compute the cross-entropy loss for a minibatch. Cross-entropy measures how well the model's predicted probability distribution matches the actual next character. A model that assigns high probability to the correct next token will have low loss.
[X,T] = mbqTrain.next();
loss = dlfeval(@nanogpt.modelLoss, bigramNet, X, T)
%[text] The expected performance of a random model is $-\\log \\frac{1}{n\_\\mathrm{vocab}$.
%[text] Compute this value for comparison.
expectedRandomLoss = -log(1/tokenizer.VocabSize)
%[text] Generate 100 tokens from the untrained model. Generation works autoregressively: take the model's output probabilities for the last position, sample a token from that distribution, append it to the context, and repeat. Before training, the output will be random garbage since the embedding table hasn't learned any character patterns yet.
numNewChars = 100;
dsValidation.MiniBatchSize = 1;
tbl = dsValidation.preview();
context = tbl.predictors;
predictFromModel(bigramNet, context, numNewChars, tokenizer)
%%
%[text] Train the model for 2000 iterations. A relatively high learning rate (1e-2) works well here because the bigram model is very simple -- just a single embedding table with no deep layers to destabilize.
numIterations = 2000;
validationFrequency = 100;
numValidationIters = 20;
learnRate = 1e-2;

bigramNet = nanogpt.train(bigramNet, mbqTrain, mbqValidation, ...
    LearnRate=learnRate, ...
    NumIterations=numIterations, ...
    ValidationFrequency=validationFrequency, ...
    NumValidationIters=numValidationIters);
%%
%[text] Use the trained model to predict 500 new tokens. The output will have learned some basic character patterns (common letter pairs, spacing), but won't form real words because the bigram model has no mechanism for tokens to share information with each other.
numNewChars = 500;
dsValidation.MiniBatchSize = 1;
tbl = dsValidation.preview();
context = tbl.predictors;
predictFromModel(bigramNet, context, numNewChars, tokenizer)
%%
%[text] ## Add Self-Attention
%[text] The bigram model's fundamental limitation is that tokens can't communicate -- token at position 5 has no idea what tokens at positions 1-4 are. We want a mechanism where each token can gather information from all tokens that came *before* it (but not after, since that would be "cheating" by looking into the future).
%[text] Self-attention solves this through a clever mechanism: every token emits three vectors computed by learned linear projections:
%[text] - Query (Q): *What am I looking for?*
%[text] - Key (K): *What do I contain?*
%[text] - Value (V): *If you find me interesting, here's the information I give you.* \
%[text] The dot product between a query and a key measures *affinity* -- how relevant is one token to another. High dot product means high affinity, so token $i$ should pay more attention to token $j$. The key insight is that attention is fundamentally just a *weighted sum*, where the weights are learned and data-dependent rather than fixed.
outputSize = tokenizer.VocabSize;
numKeyChannels = 32;
%[text] Build the attention head step by step. First, compute attention scores as $QK^T / \\sqrt{d\_k}$. The scaling by $1/\\sqrt{d\_k}$ is critical: without it, when `numKeyChannels` is large, dot products grow large in magnitude, pushing softmax into extremely peaked (nearly one-hot) regions where gradients become vanishingly small. Scaling keeps the variance of the dot products near 1, preserving healthy gradient flow.
layers = [
    functionLayer(@(x,y) nanogpt.batchmtimes(x,"none",y,"transpose"),Name="mtimes_KQ")
    functionLayer(@(x) x/sqrt(numKeyChannels), Name="scale")];
%[text] Apply a causal mask to prevent the network from "cheating" by looking at future tokens. The mask sets the upper triangle of the attention matrix to $-\\infty$ before softmax. Since $\\mathrm{softmax}(-\\infty) = 0$, future positions get zero attention weight while past positions get properly normalized weights. This is what makes the attention "decoder-style" or "autoregressive."
layers = [layers
    functionLayer(@(x) nanogpt.applyCausalMask(x),Name="mask")];
%[text] Apply softmax to get attention weights (a valid probability distribution), then multiply by the values. Each token's output is a weighted combination of all previous tokens' value vectors, where the weights reflect learned relevance.
layers = [layers
    softmaxLayer
    functionLayer(@(x,y) nanogpt.batchmtimes(x,"none",y,"none"),Name="mtimes_KQV")
    fullyConnectedLayer(outputSize,Name="fc_out")];

attentionNet = dlnetwork(layers,Initialize=false);

% Add the keys, queries and values
queryLayer = fullyConnectedLayer(numKeyChannels,Name="fc_queries");
keyLayer = fullyConnectedLayer(numKeyChannels,Name="fc_keys");
valueLayer = fullyConnectedLayer(numKeyChannels,Name="fc_values");
attentionNet = addLayers(attentionNet,queryLayer);
attentionNet = addLayers(attentionNet,keyLayer);
attentionNet = addLayers(attentionNet,valueLayer);
% Connect them up
attentionNet = connectLayers(attentionNet,"fc_queries","mtimes_KQ/in1");
attentionNet = connectLayers(attentionNet,"fc_keys","mtimes_KQ/in2");
attentionNet = connectLayers(attentionNet,"fc_values","mtimes_KQV/in2");
%[text] Plot the network to view its structure
plot(attentionNet)
%[text] Connect this attention head up to the bigram network, with the following modifications:
%[text] 1. Change the number of embedding dimensions from vocabSize to 32
%[text] 2. Add a fully connected layer after the wordEmbeddingLayer to change the number of channels to match the vocab size
%[text] 3. Move the softmax layer to after the attention head \
numChannels = 1;
embeddingDimension = 32;
inLayers = [sequenceInputLayer(numChannels,Name="input"), ...
    wordEmbeddingLayer(embeddingDimension,tokenizer.VocabSize), ...
    additionLayer(2, Name="emb_add"), ...
    fullyConnectedLayer(tokenizer.VocabSize, Name="fc_in"), ...
];
attentionNet = addLayers(attentionNet,inLayers);
attentionNet = connectLayers(attentionNet,"fc_in","fc_queries");
attentionNet = connectLayers(attentionNet,"fc_in","fc_keys");
attentionNet = connectLayers(attentionNet,"fc_in","fc_values");
attentionNet = addLayers(attentionNet,softmaxLayer(Name="soft_out"));
attentionNet = connectLayers(attentionNet,"fc_out","soft_out");

plot(attentionNet)
%[text] Attention operates on a *set*, not a sequence -- if you shuffled the input tokens, the attention computation would produce the same result (just reordered). This means without positional information, the model has no idea whether a token is at the start or end of a sentence. Position embeddings fix this by learning a unique vector for each position (0 through `subsequenceLength`-1) and adding it to the token embedding. The model can then learn position-dependent patterns like "the first word of a sentence is often capitalized."
attentionNet = attentionNet.addLayers(positionEmbeddingLayer(embeddingDimension, subsequenceLength, Name="posembed"));
attentionNet = attentionNet.connectLayers("input", "posembed");
attentionNet = attentionNet.connectLayers("posembed", "emb_add/in2");
attentionNet = attentionNet.initialize();

plot(attentionNet)
analyzeNetwork(attentionNet)
%%
%[text] Train the network for 5000 iterations. The learning rate is reduced to 1e-3 (from 1e-2 for the bigram model) because deeper networks with more parameters need smaller, more careful updates to train stably.
numIterations = 5000;
validationFrequency = 250;
numValidationIters = 20;
learnRate = 1e-3;

attentionNet = nanogpt.train(attentionNet, mbqTrain, mbqValidation, ...
    LearnRate=learnRate, ...
    NumIterations=numIterations, ...
    ValidationFrequency=validationFrequency, ...
    NumValidationIters=numValidationIters);
%%
%[text] Generate 500 tokens from the trained network.
numNewChars = 500;
dsValidation.MiniBatchSize = 1;
tbl = dsValidation.preview();
context = tbl.predictors;
predictFromModel(attentionNet, context, numNewChars, tokenizer)
%%
%[text] ## Multi-head Attention
%[text] Currently the network uses a single self-attention head -- one "communication channel." Using multiple heads running in parallel allows the model to attend to different types of relationships simultaneously. For example, one head might learn to attend to the immediately preceding vowel, another might track the start of the current word, and another might focus on sentence boundaries. Each head operates on a smaller dimension (`embeddingDimension / numHeads`), and their outputs are concatenated back together.
%[text] Using the `selfAttentionLayer` from the Deep Learning Toolbox we can straightforwardly add multiple heads.
numChannels = 1;
embeddingDimension = 32;
numHeads = 4;
numKeyChannels = embeddingDimension;

multiHeadAttentionNet = dlnetwork([ ...
    sequenceInputLayer(numChannels, Name="input"), ...
    wordEmbeddingLayer(embeddingDimension,tokenizer.VocabSize), ...
    additionLayer(2, Name="emb_add"), ...
    fullyConnectedLayer(tokenizer.VocabSize), ...
    selfAttentionLayer(numHeads, numKeyChannels, AttentionMask="causal"), ...
    softmaxLayer], Initialize=false);
multiHeadAttentionNet = multiHeadAttentionNet.addLayers(positionEmbeddingLayer(embeddingDimension, subsequenceLength, Name="posembed"));
multiHeadAttentionNet = multiHeadAttentionNet.connectLayers("input", "posembed");
multiHeadAttentionNet = multiHeadAttentionNet.connectLayers("posembed", "emb_add/in2");
multiHeadAttentionNet = multiHeadAttentionNet.initialize();
%[text] Train the network for 5000 iterations and generate 500 new tokens.
numIterations = 5000;
validationFrequency = 250;
numValidationIters = 20;
learnRate = 1e-3;

multiHeadAttentionNet = nanogpt.train(multiHeadAttentionNet, mbqTrain, mbqValidation, ...
    LearnRate=learnRate, ...
    NumIterations=numIterations, ...
    ValidationFrequency=validationFrequency, ...
    NumValidationIters=numValidationIters);
numNewChars = 500;
dsValidation.MiniBatchSize = 1;
tbl = dsValidation.preview();
context = tbl.predictors;
predictFromModel(multiHeadAttentionNet, context, numNewChars, tokenizer)
%%
%[text] The transformer architecture has two distinct phases: *communication* (attention, where tokens gather information from each other) and *computation* (feedforward, where each token independently processes what it gathered). Adding a fully connected layer with a ReLU activation after attention provides this computation phase. Think of it as: attention lets tokens "look around" at each other, and the feedforward layer lets each token "think about" what it saw.
multiHeadAttentionNet2 = dlnetwork([ ...
    sequenceInputLayer(numChannels, Name="input"), ...
    wordEmbeddingLayer(embeddingDimension,tokenizer.VocabSize), ...
    additionLayer(2, Name="emb_add"), ...
    fullyConnectedLayer(tokenizer.VocabSize), ...
    selfAttentionLayer(numHeads, numKeyChannels, AttentionMask="causal"), ...
    fullyConnectedLayer(tokenizer.VocabSize), ...
    reluLayer, ...
    softmaxLayer], Initialize=false);
multiHeadAttentionNet2 = multiHeadAttentionNet2.addLayers(positionEmbeddingLayer(embeddingDimension, subsequenceLength, Name="posembed"));
multiHeadAttentionNet2 = multiHeadAttentionNet2.connectLayers("input", "posembed");
multiHeadAttentionNet2 = multiHeadAttentionNet2.connectLayers("posembed", "emb_add/in2");
multiHeadAttentionNet2 = multiHeadAttentionNet2.initialize();

multiHeadAttentionNet2 = nanogpt.train(multiHeadAttentionNet2, mbqTrain, mbqValidation, ...
    LearnRate=learnRate, ...
    NumIterations=numIterations, ...
    ValidationFrequency=validationFrequency, ...
    NumValidationIters=numValidationIters);
%%
tbl = dsValidation.preview();
context = tbl.predictors;
predictFromModel(multiHeadAttentionNet2, context, numNewChars, tokenizer)
%%
%[text] ## Making the network deeper
%[text] Make the network deeper by repeating attention blocks. However, simply stacking blocks creates a deep network that trains poorly -- gradients vanish as they flow backward through many layers, so early layers stop learning. Two techniques solve this:
%[text] **Residual (skip) connections** create a "highway" for gradients. Instead of `x = sublayer(x)`, we compute `x = x + sublayer(x)`. During backpropagation, gradients flow freely through the addition, bypassing the potentially problematic sublayer transformations. The sublayers become incremental updates to a "residual stream" rather than wholesale replacements.
%[text] **Layer normalization** normalizes activations to have zero mean and unit variance across the embedding dimension (per token, per sample). This prevents activations from growing too large or too small as data passes through many layers, stabilizing training. This block uses "post-norm" placement (normalize *after* the sublayer and residual addition).
deeperNet = dlnetwork([ ...
    sequenceInputLayer(numChannels, Name="input"), ...
    wordEmbeddingLayer(embeddingDimension,tokenizer.VocabSize), ...
    additionLayer(2, Name="emb_add"), ...
    fullyConnectedLayer(tokenizer.VocabSize, Name="emb_fc")], Initialize=false);
deeperNet = deeperNet.addLayers(positionEmbeddingLayer(embeddingDimension, subsequenceLength, Name="posembed"));
deeperNet = deeperNet.connectLayers("input", "posembed");
deeperNet = deeperNet.connectLayers("posembed", "emb_add/in2");
lastLayerName = "emb_fc";

%[text] Each attention block contains: self-attention (communication) $\\rightarrow$ residual add $\\rightarrow$ layer norm $\\rightarrow$ feedforward with 4$\\times$ expansion (computation) $\\rightarrow$ residual add $\\rightarrow$ layer norm. The 4$\\times$ expansion in the feedforward layer (e.g. `vocabSize` $\\rightarrow$ `4*vocabSize` $\\rightarrow$ `vocabSize`) follows the original Transformer paper and gives the network more capacity for per-token computation.
numAttnBlocks = 3;
for ii = 1:numAttnBlocks
    attnBlock = attentionBlock(numHeads, numKeyChannels, tokenizer.VocabSize, ii);
    deeperNet = deeperNet.addLayers(attnBlock);
    deeperNet = deeperNet.connectLayers(lastLayerName, "attn_sa_"+ii);
    % Create skip connections (bypass attention and feedforward sublayers)
    deeperNet = deeperNet.connectLayers(lastLayerName, "attn_add1"+ii+"/in2");
    deeperNet = deeperNet.connectLayers("attn_norm1_"+ii, "attn_add2"+ii+"/in2");
    lastLayerName = "attn_norm2_"+ii;
end

deeperNet = deeperNet.addLayers([geluLayer(Name="last_gelu"), softmaxLayer]);
deeperNet = deeperNet.connectLayers(lastLayerName, "last_gelu");
deeperNet = deeperNet.initialize();
%[text] Train the network for 10000 iterations and predict 500 new tokens.
numIterations = 10000;
validationFrequency = 500;
deeperNet = nanogpt.train(deeperNet, mbqTrain, mbqValidation, ...
    LearnRate=learnRate, ...
    NumIterations=numIterations, ...
    ValidationFrequency=validationFrequency, ...
    NumValidationIters=numValidationIters);
%%
tbl = dsValidation.preview();
context = tbl.predictors;
predictFromModel(deeperNet, context, numNewChars, tokenizer)
%%
%[text] ### Build a larger model for best performance
%[text] Scale up the architecture: wider embeddings (384 vs. 32), more heads (6 vs. 4), and more blocks (6 vs. 3). Remarkably, the architecture is identical to what GPT-2 and GPT-3 use -- only the hyperparameter values differ. GPT-2 uses `embeddingDimension=768`, `numHeads=12`, `numBlocks=12`, and `subsequenceLength=1024`.
%[text] **Dropout** is added as regularization to prevent overfitting. During training, it randomly zeros out 20% of activations, forcing the network to spread knowledge across multiple pathways rather than relying on any single feature. At inference time, dropout is disabled. Dropout is applied in four places: after the embedding sum, on the attention weights (inside `selfAttentionLayer`), after the attention output, and after the feedforward layer.
%[text] This block also switches to the **pre-norm** formulation: `x = x + sublayer(LayerNorm(x))` instead of the post-norm `x = LayerNorm(x + sublayer(x))` used earlier. Pre-norm is now the standard in modern transformers because it is generally easier to train -- the residual stream stays unnormalized, which helps gradient flow.
embeddingDimension = 384;
numKeyChannels = 384;
numHeads = 6;
dropout = 0.2;
numAttnBlocks = 6;
miniBatchSize = 64;
%[text] Increase the context window from 8 to 256 tokens, giving the model much more context for each prediction.
subsequenceLength = 256;
dsTrain = nanogpt.SequenceForecastingDatastore(dataTrain, subsequenceLength, tokenizer.VocabSize);
dsValidation = nanogpt.SequenceForecastingDatastore(dataValidation, subsequenceLength, tokenizer.VocabSize);
mbqTrain = minibatchqueue(dsTrain, ...
    MinibatchSize=miniBatchSize, ...
    MiniBatchFcn=@(X,T) preprocessMinibatch(X,T), ...
    MiniBatchFormat={'BTC', 'BTC'});
mbqValidation = minibatchqueue(dsValidation, ...
    MinibatchSize=miniBatchSize, ...
    MiniBatchFcn=@(X,T) preprocessMinibatch(X,T), ...
    MiniBatchFormat={'BTC', 'BTC'});

deepestNet = dlnetwork([ ...
    sequenceInputLayer(numChannels, Name="input"), ...
    wordEmbeddingLayer(embeddingDimension,tokenizer.VocabSize), ...
    additionLayer(2, Name="emb_add"), ...
    dropoutLayer(dropout, Name="emb_drop")], Initialize=false);
deepestNet = deepestNet.addLayers(positionEmbeddingLayer(embeddingDimension, subsequenceLength, Name="posembed"));
deepestNet = deepestNet.connectLayers("input", "posembed");
deepestNet = deepestNet.connectLayers("posembed", "emb_add/in2");
lastLayerName = "emb_drop";

for ii = 1:numAttnBlocks
    attnBlock = preNormAttentionBlock(numHeads, numKeyChannels, embeddingDimension, ii, Dropout=dropout);
    deepestNet = deepestNet.addLayers(attnBlock);
    deepestNet = deepestNet.connectLayers(lastLayerName, "attn_norm1_"+ii);
    % Create skip connections (bypass norm+sublayer)
    deepestNet = deepestNet.connectLayers(lastLayerName, "attn_add1"+ii+"/in2");
    deepestNet = deepestNet.connectLayers("attn_add1"+ii, "attn_add2"+ii+"/in2");
    lastLayerName = "attn_add2"+ii;
end
%[text] A final layer norm is applied after all blocks (needed in the pre-norm formulation since the last block's output is unnormalized), then a "language model head" projects from the embedding dimension back to vocabulary size to produce logits over all possible next tokens.
deepestNet = deepestNet.addLayers([layerNormalizationLayer(Name="final_norm"), fullyConnectedLayer(tokenizer.VocabSize, Name="lm_head"), softmaxLayer]);
deepestNet = deepestNet.connectLayers(lastLayerName, "final_norm");
deepestNet = deepestNet.initialize();
%%
%[text] Train the network for up to 5000 iterations with several modern training techniques:
%[text] - **Learning rate warmup**: Start with a very small learning rate and linearly ramp up over the first 100 iterations. This prevents large, destabilizing updates early on when the model's gradients are noisy and unreliable.
%[text] - **Cosine decay**: After warmup, smoothly reduce the learning rate following a cosine curve down to `minLearnRate`. This lets the model make large exploratory updates early and fine-grained refinements later.
%[text] - **Gradient clipping** (`maxGradientNorm=1.0`): If the global gradient norm exceeds 1.0, scale all gradients down proportionally. This prevents rare large gradients from causing catastrophic parameter updates.
%[text] - **Weight decay** (0.1): A form of regularization that gently pushes weights toward zero, preventing any single weight from growing too large. Unlike L2 regularization, AdamW applies weight decay *after* the adaptive learning rate step (decoupled weight decay).
%[text] - **Early stopping** (`validationPatience=10`): Stop training if validation loss hasn't improved for 10 consecutive checks, preventing wasted computation and overfitting. \
numIterations = 5000;
validationFrequency = 250;
numValidationIters = 200;
validationPatience = 10;
learnRate = 1e-3;
minLearnRate = 1e-4;
warmupIterations = 100;
maxGradientNorm = 1.0;
doTrain = false;

if doTrain %#ok<*UNRCH>
    deepestNet = nanogpt.train(deepestNet, mbqTrain, mbqValidation, ...
        LearnRate=learnRate, ...
        MinLearnRate=minLearnRate, ...
        WarmupIterations=warmupIterations, ...
        MaxGradientNorm=maxGradientNorm, ...
        WeightDecay=0.1, ...
        Beta2=0.99, ...
        NumIterations=numIterations, ...
        ValidationFrequency=validationFrequency, ...
        NumValidationIters=numValidationIters, ...
        ValidationPatience=validationPatience, ...
        Verbose=true);
    save(fullfile(pwd, "models", "nanogpt_"+ corpusFileName + ".mat"), "deepestNet")
else
    load(fullfile(pwd, "models", "nanogpt_"+ corpusFileName + ".mat"), "deepestNet")
end
%%
numNewChars = 1000;
dsValidation.MiniBatchSize = 1;
tbl = dsValidation.preview();
context = tbl.predictors;
predictFromModel(deepestNet, context, numNewChars, tokenizer)
%%
%[text] Export the Live Script to HTML.
% if ~isfolder("results"), mkdir("results"); end
% export("BuildNanoGPT.m", fullfile("results", "BuildNanoGPT_" + corpusFileName + ".html"));
%%
function [X,T] = preprocessMinibatch(dataX, dataT)
X = cat(1, dataX{:});

T = cat(1, dataT{:});
T = onehotencode(T, 3);
end

function varargout = predictFromModel(net, context, numChars, tokenizer)
streamOutput = nargout == 0;
currentContext = context;
numBatches = size(context, 1);
newChars = repmat('a', [numBatches numChars]);
for ii = 1:numChars
    prediction = net.predict(currentContext,InputDataFormats="BTC",OutputDataFormats="CBT"); % CBT dlarray

    lastPrediction = prediction(:,:,end); % 1BT dlarray
    predictedToken = sampleFromLogits(lastPrediction); % 1B1 categorical
    currentContext = cat(2, currentContext(:,2:end), predictedToken);

    nextToken = squeeze(predictedToken); %B1 double
    nextChar = tokenizer.tok2char(nextToken);
    newChars(:,ii) = nextChar;
    if streamOutput
        fprintf("%s", nextChar);
        drawnow;
    end
end
if streamOutput
    fprintf("\n");
else
    varargout{1} = newChars;
end
end

function token = sampleFromLogits(logits)
% Given a vector of probabilities, sample a token with the probability
% weight given by the elements of that vector

population = 1:size(logits,1);
numBatches = size(logits,2);
token = zeros(1,numBatches);
for ii = 1:numBatches
    token(:,ii) = randsample(population, 1, true, gather(logits(:,ii)));
end
end

function layers = attentionBlock(numHeads, numKeyChannels, vocabSize, index, opts)
arguments
    numHeads
    numKeyChannels
    vocabSize
    index
    opts.Dropout = 0
end
layers = [...
    selfAttentionLayer(numHeads, numKeyChannels, AttentionMask="causal", Name="attn_sa_"+index), ...
    dropoutLayer(opts.Dropout, Name="attn_drop1_"+index), ...
    additionLayer(2, Name="attn_add1"+index), ...
    layerNormalizationLayer(Name="attn_norm1_"+index), ...
    fullyConnectedLayer(4*vocabSize, Name="attn_fc_"+index), ...
    geluLayer(Name="attn_act_"+index), ...
    dropoutLayer(opts.Dropout, Name="attn_drop2_"+index), ...
    fullyConnectedLayer(vocabSize), ...
    additionLayer(2, Name="attn_add2"+index), ...
    layerNormalizationLayer(Name="attn_norm2_"+index)];
end

function layers = preNormAttentionBlock(numHeads, numKeyChannels, dim, index, opts)
% Pre-norm transformer block (norm before sublayer, skip bypasses norm)
arguments
    numHeads
    numKeyChannels
    dim
    index
    opts.Dropout = 0
end
layers = [...
    layerNormalizationLayer(Name="attn_norm1_"+index), ...
    selfAttentionLayer(numHeads, numKeyChannels, AttentionMask="causal", DropoutProbability=opts.Dropout, Name="attn_sa_"+index), ...
    dropoutLayer(opts.Dropout, Name="attn_drop1_"+index), ...
    additionLayer(2, Name="attn_add1"+index), ...
    layerNormalizationLayer(Name="attn_norm2_"+index), ...
    fullyConnectedLayer(4*dim, Name="attn_fc1_"+index), ...
    geluLayer(Name="attn_act_"+index), ...
    fullyConnectedLayer(dim, Name="attn_fc2_"+index), ...
    dropoutLayer(opts.Dropout, Name="attn_drop2_"+index), ...
    additionLayer(2, Name="attn_add2"+index)];
end
%%
%[text] *Copyright 2026 The MathWorks, Inc.*

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
%[control:dropdown:903b]
%   data: {"defaultValue":"\"tinyshakespeare\"","itemLabels":["\"tinyshakespeare\"","\"quijote\""],"items":["\"tinyshakespeare\"","\"quijote\""],"label":"corpusFileName","run":"Section"}
%---
