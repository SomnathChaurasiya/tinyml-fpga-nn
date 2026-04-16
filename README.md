# FPGA Neural Network (TinyML)

## Overview
2-layer FCNN on FPGA (Verilog) for MNIST digit classification.

## Architecture
- FC1: 784 → 64 (ReLU)
- FC2: 64 → 10
- Argmax → predicted digit

## Result
Predicted digit: 3

## How to run
- Vivado XSIM
- Run `tb/neuron_tb.v`

## Repo contents
- rtl/: RTL (neuron)
- tb/: testbench
- data/: input/weights/bias (text)
- docs/: report

## Author
Somnath Chaurasiya
