## Efficient Keyword Spotting for Embedded Systems

# Overview
This repository demonstrates an efficient keyword spotting system tailored for STM32L4 microcontrollers, 
balancing accuracy and speed for real-time audio processing in embedded systems. 
Deployed on an STM32L4 running at 80Mhz, achieves an inference latency of 190ms, suitable for continuous monitoring applications.

**Demo** For reference, the model can be tested in the browser [here](https://www.nikbamert.com/browser_demo_inference.html).

# Key Features
- Optimized for Embedded Systems: Designed for real-time performance on STM32L4 microcontrollers.
- Resource-Efficient: Achieves high efficiency with limited computational resources.
- Model Design: Utilizes a modified M5 model, processing raw waveforms for enhanced responsiveness.
- Live Demo: Experience the model in action via our browser demo.

# Repository Structure
- `model_training`: Contains Pytorch Lightning training code.
- `browser_inference`: Includes browser-based demo inference code. Try it here.
- `stm32_inference`: Features STM32-specific inference engine with firmware for B-L475-IOT01A board.

# Model Specifications
- Dataset: Recognizes 35 keywords from the speech commands dataset.
- Inference Time: Approximately 190ms at 80Mhz.
- Memory Usage: Consumes about 60Kb RAM.

# Getting Started
- The python requirements are managed with `poetry`. They are installed with `cd model_training && poetry install`.
- The stm32 code requires the arm gcc: `arm-none-eabi-gcc`. Build the code with `cd stm32_inference && make`.
    - A firmware binary is available at `stm32_inference/build/speechmodel_code.bin`.
- Includes a no-frills browser inference engine in `browser_inference/browser_demo_inference.html` 

