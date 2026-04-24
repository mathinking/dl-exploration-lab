# AI Verification: Proving Neural Network Robustness

This exercise demonstrates how to verify that neural networks behave correctly using formal methods from the AI Verification Library. Using the ACAS Xu (Airborne Collision Avoidance System for unmanned aircraft) neural networks, the exercise progresses from exploring network behavior, to formally proving local robustness, to verifying global stability across the entire operational design domain.

Based on the example [Verify and Deploy ACAS Xu Neural Networks](https://www.mathworks.com/help/deeplearning/ug/verify-and-deploy-acas-xu-neural-networks.html).

## Pipeline Overview

| Stage | Topic | Description |
|-------|-------|-------------|
| 1 | **Explore ACAS Xu** | Load a collision avoidance network, simulate an encounter trajectory, and observe advisory transitions as aircraft separation changes |
| 2 | **Verify Local Robustness** | Use CROWN and alpha-CROWN formal verification to prove the network is robust to input perturbations at a single operating point |
| 3 | **Verify Global Stability** | Partition the full operational design domain using adaptive mesh refinement, verify each region, and quantify coverage |

## About the Data

**ACAS Xu Neural Networks** — A set of 45 fully connected networks from the Airborne Collision Avoidance System for unmanned aircraft. Each network takes 5 inputs (aircraft geometry and velocities) and outputs one of 5 steering advisories. Downloaded automatically by `helper.setupModels`.

## Required Products

- [MATLAB](https://www.mathworks.com/products/matlab.html) R2026a or later
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)

## Required Add-Ons

Install the following from the MATLAB Add-On Explorer:

- [AI Verification Library for Deep Learning Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/ai-verification-library-for-deep-learning-toolbox)

## Optional Products

- [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html) — Enables GPU acceleration for faster verification

## Getting Started

1. Navigate to this folder in MATLAB
2. Run `helper.setupModels` to download the ACAS Xu neural networks
3. Open `AIVerification.m` as a Live Script and run section by section

## Results

View the full pre-generated HTML report:

* [AIVerification](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/d_AI-Verification/results/AIVerification.html)

## References

- [Verify and Deploy ACAS Xu Neural Networks](https://www.mathworks.com/help/deeplearning/ug/verify-and-deploy-acas-xu-neural-networks.html)
- [verifyNetworkRobustness](https://www.mathworks.com/help/deeplearning/ref/verifynetworkrobustness.html)
- [estimateNetworkOutputBounds](https://www.mathworks.com/help/deeplearning/ref/estimatenetworkoutputbounds.html)

*Copyright 2026 The MathWorks, Inc.*
