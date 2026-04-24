# Image Classification with Convolutional Neural Networks

Build, train, and evaluate CNNs for handwritten digit classification in MATLAB&reg;. Starting from a minimal single-layer CNN and progressively improving the architecture, you will see how design choices &mdash; learning rate, depth, batch normalization, and pooling &mdash; affect classification accuracy.

## Requirements

| Product | Required |
|---|---|
| [MATLAB&reg;](https://www.mathworks.com) R2023b+ | Yes |
| [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html) | Yes |

## Getting Started

Open `ImageClassification.m` and run it section by section. The script uses the built-in digit dataset that ships with Deep Learning Toolbox &mdash; no additional data downloads are required.

The exercise includes interactive sections where you build a network in [Deep Network Designer](https://www.mathworks.com/help/deeplearning/ref/deepnetworkdesigner-app.html), explore networks of different depths and complexities, tune the learning rate, and evaluate classification performance.

## Solution

The complete solution with pre-generated results is available here:

* [ImageClassification &mdash; Solution](https://mathinking.github.io/dl-exploration-lab/2-Image-Classification/results/ImageClassification.html)

## Pipeline Overview

| Stage | Topic | Key Concepts |
|---|---|---|
| Define Architecture | Minimal CNN | `imageInputLayer`, `convolution2dLayer`, `reluLayer`, Deep Network Designer |
| Fine Tuning 1 | Learning Rate | Impact of learning rate on convergence and accuracy |
| Fine Tuning 2 | Deeper CNN | Batch normalization, max pooling, hierarchical features |
| Evaluation | Predictions & Confusion Matrix | `minibatchpredict`, `scores2label`, `confusionchart` |

## References

1. *Create Simple Deep Learning Network for Classification*. Available at https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-classification-network.html.
2. *Deep Network Designer*. Available at https://www.mathworks.com/help/deeplearning/ref/deepnetworkdesigner-app.html.

---

Copyright 2024-2026 The MathWorks, Inc.
