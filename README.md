# Efficient Keyword Spotting for Embedded Systems

This repository demonstrates an efficient keyword spotting system tailored for STM32L4 microcontrollers, 
balancing accuracy and speed for real-time audio processing in embedded systems. 
Deployed on an STM32L4 running at 80Mhz it recognizes [35 different keywords](model_training/dataset.py#L9-L44) and achieves an inference latency of 190ms, 
suitable for continuous monitoring applications.

**Demo** For reference, the model can be tested in the browser [here](https://www.nikbamert.com/browser_demo_inference.html).

## Model Specifications
- Utilizes a modified [M5 model](https://arxiv.org/abs/1610.00087), processing raw waveforms (no spectrogram).
- Dataset: Recognizes 35 keywords from the [speech commands dataset](https://arxiv.org/abs/1804.03209).
- Inference Time (Cortex M4): ~ 190ms at 80Mhz (Cortex M4).
- Inference Time (Browser):  ~ 1-5ms depending on device
- Memory Usage (Cortex M4): Consumes about 60Kb RAM.

## Repository Structure
- `model_training`: Contains Pytorch Lightning training code.
- `browser_inference`: Includes browser-based demo inference code. Try it here.
- `stm32_inference`: Features STM32-specific inference engine with firmware for [B-L475-IOT01A](https://www.st.com/en/evaluation-tools/b-l475e-iot01a.html) board.

## Getting Started
- The python requirements are managed with `poetry`. They are installed with `cd model_training && poetry install`.
- The stm32 code requires the arm gcc: `arm-none-eabi-gcc`. Build the code with `cd stm32_inference && make`.
    - A firmware binary is available at `stm32_inference/build/speechmodel_code.bin`.
- Includes a no-frills browser inference engine in `browser_inference/browser_demo_inference.html` 


## Model accuracy / inference time tradeoff
| Model | val acc. | stm32 inference time [ms] | MFLOP | kParams |
| ------------- | ------------- | ---- | ---- | ---- |
| M5 (c=32, k=80) | 0.853 | 603 | 3.8 | 166 |
| M5 (c=16, k=80) | 0.79 | -  |  - | - |
| M5 (c=16, k=40) | 0.863 | 595 |  2.4 | 99 |
| M5 (c=16, k=20) | 0.852 | 246 |  1.8 | 98 |
| M5 (c=16, k=10) | 0.812 | 180 |  1.6 | 97 |

