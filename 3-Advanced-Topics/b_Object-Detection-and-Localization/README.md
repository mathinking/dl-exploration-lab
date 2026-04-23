# Object Detection and Pose Estimation: Satellite Pose Estimation

Estimate the 6-DoF pose (orientation and position) of a CubeSat from 2D images using deep learning and classical computer vision in MATLAB&reg;. The pipeline combines a YOLOv4-based object detector, a keypoint regression network, and a Perspective-n-Point (PnP) solver to go from a raw image to a full 3D pose estimate.

Based on the [AI CubeSat Pose Estimation Workshop](https://github.com/mathworks/AI-CubeSat-Pose-Estimation-Workshop) developed in collaboration with Stanford University's Space Rendezvous Laboratory.

## Requirements

| Product | Required |
|---|---|
| [MATLAB&reg;](https://www.mathworks.com) R2023b+ | Yes |
| [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html) | Yes |
| [Computer Vision Toolbox&trade;](https://www.mathworks.com/products/computer-vision.html) | Yes |
| [Image Processing Toolbox&trade;](https://www.mathworks.com/products/image-processing.html) | Yes |
| [Aerospace Toolbox&trade;](https://www.mathworks.com/products/aerospace-toolbox.html) | Yes |

## Getting Started

Open `SatellitePoseEstimation.m` and run it section by section. The script walks through each stage of the pose estimation pipeline using pre-trained models &mdash; no training is required.

Pre-trained models are stored in `models/` and helper functions are organized in the `+helper` namespace package.

## Results

View the full pre-generated HTML report:

* [SatellitePoseEstimation](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/b_Object-Detection-and-Localization/results/SatellitePoseEstimation.html)

![Fly-By Pose Estimation](results/FlyByAnnotated.gif)

## Pipeline Overview

| Stage | Method | Output |
|---|---|---|
| Object Detection | YOLOv4 network ([pre-trained](https://github.com/tpark94/speed-ue-cube-baseline/blob/main/trainODN.m)) | Bounding box around satellite |
| Keypoint Detection | Convolutional regression network ([pre-trained](https://github.com/tpark94/speed-ue-cube-baseline/blob/main/trainKRN.m)) | 11 keypoint locations in 2D |
| Pose Estimation | PnP solver (`estworldpose`) | Camera position and orientation in 3D |

## Dataset

The demo uses a subset of the [SPEED-UE-Cube](https://purl.stanford.edu/hw812wb1641) dataset &mdash; synthetically rendered images of a 3U CubeSat as seen from an approaching spacecraft camera. Ground truth includes quaternion orientation and translation for each image.

## References

1. Park, T.H., Ahmed, Z., et al. (2024). *Spacecraft Pose Estimation Dataset of a 3U CubeSat using Unreal Engine (SPEED-UE-Cube)*. Available at https://purl.stanford.edu/hw812wb1641.
2. Ahmed, Z., Park, T.H., et al. *SPEED-UE-Cube: A Machine Learning Dataset for Autonomous, Vision-Based Spacecraft Navigation*. 46th Rocky Mountain AAS GN&C Conference, 2024.
3. Stanford GitHub repository for MATLAB deep learning training: https://github.com/tpark94/speed-ue-cube-baseline.

---

Copyright 2024-2026 The MathWorks, Inc.
