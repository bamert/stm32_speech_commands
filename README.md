# STM32 Speech Commands

The code in this repo demonstrates keyword spotting ("yes", "off", "on", "up", "right", ..) with very limited compute on STM32L4 microcontrollers based on the speech commands dataset ( [paper](https://arxiv.org/abs/1804.03209) ).
The model recognizes 35 different [keywords](model-training/dataset.py) and runs at 4+ inferences per second on the target STM32L4 device.
Contrary to other approaches for this dataset, the model in this implementation does not aim to be as accurate as possible. Instead, we want to keep inference time low given the
constrained embedded environment, while still achieving reasonable accuracy. 

For reference, the model can be run and tested out in the browser [here](https://www.nikbamert.com/browser_demo_inference.html).

# Repo structure
- `model_training` Pytorch Lightning training code. 
- `browser_inference` Demo inference code for the browser. Also served [here](https://www.nikbamert.com/browser_demo_inference.html)
- `stm32_inference` STM32 inference engine with firmware image for [B-L475-IOT01A](https://www.st.com/en/evaluation-tools/b-l475e-iot01a.html) board by ST.

# Model specifics
The model is a modified version of the [M5](https://arxiv.org/abs/1610.00087) model by Dai et al. and operates directly on the raw waveform rather than using a spectrogram / MFC features.

# Inference engine 
One forward pass on the STM32L4 takes about 180ms at 80Mhz. Ram usage is around 60Kb. 
The inference engine for both browser and stm32 record audio at 8kHz and run inference on overlapping 1sec samples about every 250ms.

