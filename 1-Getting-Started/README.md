# Deep Learning in 6 Lines of Code

Classify images using a pre-trained convolutional neural network in MATLAB&reg;. This introductory exercise shows how to load a pre-trained CNN (AlexNet), pass an image through it, and obtain a classification label &mdash; all in just six lines of code.

## Requirements

| Product | Required |
|---|---|
| [MATLAB&reg;](https://www.mathworks.com) R2023b+ | Yes |
| [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html) | Yes |

## Getting Started

Open `GettingStarted.m` and run it section by section. The script uses the built-in `peppers.png` image that ships with MATLAB &mdash; no additional data downloads are required.

## Solution

The complete solution with pre-generated results is available here:

* [GettingStarted](https://mathinking.github.io/dl-exploration-lab/1-Getting-Started/results/GettingStarted.html)

## Pipeline Overview

| Stage | Topic | Key Concepts |
|---|---|---|
| Load Pre-trained CNN | AlexNet | `imagePretrainedNetwork`, `analyzeNetwork`, pre-trained models |
| Classify Image | Image classification | `imread`, `imresize`, `predict`, `scores2label` |

## References

1. *Pretrained Deep Neural Networks*. Available at https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html.
2. *AlexNet*. Available at https://www.mathworks.com/help/deeplearning/ref/alexnet.html.

---

Copyright 2024-2026 The MathWorks, Inc.
