# Deep Learning Exploration Lab

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=mathinking/dl-exploration-lab)

A hands-on workshop for learning deep learning with MATLAB, progressing from image classification fundamentals to advanced topics including time-series modeling, object detection, language modeling, and AI verification.

## Exercises

| # | Exercise | Description | Key Concepts | Results |
|---|----------|-------------|--------------|---------|
| 1 | [Getting Started](1-Getting-Started/) | Classify an image using a pre-trained AlexNet in 6 lines of code | Pre-trained CNNs, `predict`, `scores2label` | [HTML](https://mathinking.github.io/dl-exploration-lab/1-Getting-Started/results/GettingStarted.html) |
| 2 | [Image Classification](2-Image-Classification/) | Build and fine-tune CNNs for handwritten digit recognition (MNIST) | CNN architecture, learning rate, batch normalization, Deep Network Designer | [HTML](https://mathinking.github.io/dl-exploration-lab/2-Image-Classification/results/ImageClassification.html) |
| 3a | [Time Series and Sequence Modeling](3-Advanced-Topics/a_Time-Series-and-Sequence-Modeling/) | Estimate battery state-of-charge using feedforward, LSTM, and monotonic networks | LSTMs, sequence modeling, physics-informed constraints | [HTML](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/a_Time-Series-and-Sequence-Modeling/results/BatterySOCEstimation.html) |
| 3b | [Object Detection and Localization](3-Advanced-Topics/b_Object-Detection-and-Localization/) | Estimate 6-DoF satellite pose from 2D images using YOLOv4 and keypoint regression | YOLOv4, keypoint detection, PnP pose estimation | [HTML](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/b_Object-Detection-and-Localization/results/SatellitePoseEstimation.html) |
| 3c | [Language Modeling](3-Advanced-Topics/c_Language-Modeling/) | Build a GPT from scratch and generate Shakespeare-style text | Transformers, self-attention, text generation | [HTML](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/c_Language-Modeling/results/BuildNanoGPT_tinyshakespeare.html) |
| 3d | [AI Verification](3-Advanced-Topics/d_AI-Verification/) | Prove neural network robustness using formal verification methods | CROWN, alpha-CROWN, adaptive mesh verification | [HTML](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/d_AI-Verification/results/AIVerification.html) |

## Getting Started

1. Open MATLAB R2023b or later (some advanced exercises require newer releases &mdash; see individual README files)
2. Navigate to any exercise folder
3. Open the main `.m` file as a Live Script and run section by section

Each exercise is self-contained with pre-trained models and helper functions organized in namespace packages. No training is required to run the exercises.

## Required Products

All exercises require [MATLAB](https://www.mathworks.com/products/matlab.html) and [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html). Advanced exercises may require additional toolboxes &mdash; see individual README files for details.

*Copyright 2024-2026 The MathWorks, Inc.*
