# Time Series and Sequence Modeling: Battery State-of-Charge Estimation

Estimate the State of Charge (SOC) of a lithium-ion battery using deep learning in MATLAB&reg;. SOC represents the remaining energy in a battery as a fraction from 0 (empty) to 1 (full) and is a critical quantity in electric vehicles, energy storage systems, and consumer electronics.

The script covers three progressively more sophisticated approaches: a feedforward baseline, a stacked LSTM that learns temporal dynamics, and monotonic neural networks that architecturally guarantee physically consistent behavior.

Based on the MathWorks examples [Train Network for Battery State of Charge Estimation](https://www.mathworks.com/help/deeplearning/ug/train-network-for-battery-state-of-charge-estimation.html) and [Battery SOC Estimation Using Monotonic Neural Networks](https://github.com/matlab-deep-learning/constrained-deep-learning/blob/main/examples/monotonic/BSOCEstimateUsingMonotonicNetworks/BatteryStateOfChargeEstimationUsingMonotonicNeuralNetworks.md).

## Requirements

| Product | Required |
|---|---|
| [MATLAB&reg;](https://www.mathworks.com) R2024a+ | Yes |
| [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html) | Yes |

## Getting Started

Open `BatterySOCEstimation.m` and run it section by section. The script uses pre-trained models by default &mdash; set `doTraining = true` in the relevant sections to train from scratch.

Pre-trained models are stored in `models/` and helper functions are organized in the `+helper` namespace package.

## Results

View the full pre-generated HTML report:

* [BatterySOCEstimation](https://mathinking.github.io/dl-exploration-lab/3-Advanced-Topics/a_Time-Series-and-Sequence-Modeling/results/BatterySOCEstimation.html)

## Pipeline Overview

| Stage | Method | Output |
|---|---|---|
| Feedforward Baseline | FFN with 1 hidden layer (3 raw features) | Independent per-timestep SOC estimate |
| Feature Engineering | FFN with 1 hidden layer (5 features incl. moving averages) | Improved per-timestep SOC estimate |
| Stacked LSTM | Two LSTM layers (256 &rarr; 128 hidden units, 3 features) | Sequence-aware SOC estimate |
| Monotonic Networks | Constrained LSTMs predicting SOC differences | Physically guaranteed monotonic SOC during charge/discharge |

## Dataset

- **LG HG2 18650**: Experimental data from lithium-ion battery cells at four temperatures (-10&deg;C, 0&deg;C, 10&deg;C, 25&deg;C) recorded during driving cycles. Five normalized features: voltage, current, temperature, and moving averages of voltage and current.

## References

1. *Train Network for Battery State of Charge Estimation*. Available at https://www.mathworks.com/help/deeplearning/ug/train-network-for-battery-state-of-charge-estimation.html.
2. *Constrained Deep Learning*. Available at https://github.com/matlab-deep-learning/constrained-deep-learning.

---

Copyright 2024-2026 The MathWorks, Inc.
