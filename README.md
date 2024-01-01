# Efficient Keyword Spotting for Embedded Systems

This repository demonstrates an efficient keyword spotting system tailored for STM32L4 microcontrollers, 
balancing accuracy and speed for real-time audio processing in embedded systems. 
Deployed on an STM32L4 running at 80Mhz it recognizes [35 different keywords](model_training/dataset.py#L9-L44) and achieves an inference latency of 190ms, 
suitable for continuous monitoring applications.

**Demo** For reference, the model can be tested in the browser [here](https://www.nikbamert.com/browser_demo_inference.html).

## Model Specifications
- Utilizes a modified [M5 model](https://arxiv.org/abs/1610.00087), processing raw waveforms (no spectrogram).
- Dataset: Recognizes 35 keywords from the [speech commands dataset](https://arxiv.org/abs/1804.03209).
- Audio sampling rate: 8kHz, 1 sec frames.
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

| Model | val acc. | pr val acc.(% rejected) | stm32 inference time [ms] | MFLOP | kParams |
| ------------- | ------------- | ---- | ---- | ---- | ---- |
| M5-c32-k80 | 86.6 | 96.9 (23.1)| 603 | 3.8 | 166 |
| M5-c16-k80 | 81.7 | 96.3 (37.4)| -  |  - | - |
| M5-c32-k40 | 87.6 | 97.2 (23.0)| 595 |  2.4 | 99 |
| M5-c32-k20 | 86.2 | 96.6 (23.8)| 246 |  1.8 | 98 |
| M5-c32-k10 | 84.5 | 96.5 (28.4)| **180** |  1.6 | 97 |

The above table shows some of the model configurations that were tried out. The first row
shows the original configuration of the [M5 model by Dai et al](https://arxiv.org/abs/1610.00087). 

The STM32 inferences engine acquires and runs inference on overlapping audio frames of 1 second length (8kHz; 8000samples)
every 250ms. This is to ensure that the longer keywords ("visual", "marvin", ..) have a higher likelihood of being fully contained
in one of the frames as opposed to being cut in half. To enable 4 inferences per second, the inference time of the model has to be under 250ms.

Experiments with a smaller kernel length for the initial 1D convolution showed that reasonable performance can also be reached with a much smaller `k=10`.
The accuracy on the validation split with this model is 84.5%. For keyword spotting applications it is more acceptable
to miss an unclear keyword rather than making a false positive prediction. For this reason we use the distance between the class with the highest and second highest probabilities
as a proxy for the confidence of the prediction. We only make a prediction if this distance is > 75%. Given this additional criterion to avoid false positives,
all models reach a post-rejection accuracy in excess of 96% on the non-rejected validation samples (pr val acc). 

The model used in the stm32 and browser inference engines above is the `M5-c32-k10`.
