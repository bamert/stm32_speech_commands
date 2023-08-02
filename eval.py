import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import SC_CLASSES
from src.models import M5
from src.test import get_likely_index, get_max_activation
import sounddevice as sd

# from torchsummary import summary
from thop import profile
import sched
import time

def run_inference(model, waveform):
    waveform = torch.tensor(waveform.reshape(1,1,-1))

    logits = model(waveform)
    pred_idx = get_likely_index(logits)

    probabilities_cpu = torch.exp(logits)
    prob_sorted, _ = torch.sort(probabilities_cpu, descending=True)
    prob_sorted = prob_sorted[0][0]

    highest_prob = prob_sorted[0]
    second_highest_prob = prob_sorted[1]
    confidence = highest_prob - second_highest_prob

    # conf = get_max_activation(logits)
    # probabilities = F.softmax(logits, dim=1)
    # confidence, predicted_class = torch.max(probabilities, dim=1)
    pred_name = int_to_label[pred_idx.item()]
    # confidence = confidence.detach().numpy()
    return confidence, pred_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Your Program Description')
    parser.add_argument('--stream', action='store_true', 
                    help='Include this flag to enable streaming')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_sample_rate = 8000
    print("Building model")

    int_to_label = {i: label for i, label in enumerate(sorted(SC_CLASSES))}
    num_classes = len(SC_CLASSES)
    model = M5(n_input=1, n_output=num_classes)

    print("Loading params")
    model.load_state_dict(torch.load("model"))
    model.eval()
    model.to(device)

    inp = torch.randn(1, 1, 8000)
    macs, params = profile(model, inputs=(inp, ))
    print(macs)
    print(params)

    samplerate = 8000  # Sample rate
    duration = 1  # Duration of recording in seconds

    if not args.stream: #interactive mode
        print("Hit enter to record a 1sec sample")
        while True:
            user_input = input()  # wait for user to press enter
            if user_input == 'q':  # if 'q' was entered
                print('Exiting program.')
                break  # finish the loop
            else:
                print("Recording audio...")
                audio_data = sd.rec(8000, samplerate=samplerate, channels=1)
                sd.wait()  # Wait for the recording to finish
                print("Processing audio...")
                confidence, pred_name = run_inference(model, audio_data)
                print(f"{confidence:.2f} {pred_name}")
    else:
        print("Listening to stream. Exit with Ctrl+C")
        # Parameters
        buffer_duration = 1.0  # Duration of the rolling buffer in seconds
        sample_rate = 8000  # Sample rate in Hz
        block_duration = 0.1  # Block duration in seconds
        #time to capture next sample. 0.25 -> 4 samples per second
        sample_delay = 0.25

        # Derived parameters
        buffer_samples = int(buffer_duration * sample_rate)
        block_samples = int(block_duration * sample_rate)

        # Initialize the rolling buffer
        rolling_buffer = np.zeros(buffer_samples)

        def callback(indata, frames, time, status):
            # This is called (from a separate thread) for each audio block.
            # Update the rolling buffer with the new audio data
            rolling_buffer[:-frames] = rolling_buffer[frames:]
            rolling_buffer[-frames:] = indata[:, 0]

        def call_f(sc):
            audio_data = rolling_buffer[-buffer_samples:]
            audio_data = np.array(audio_data).astype(np.float32)

            confidence, pred_name = run_inference(model, audio_data)
            if confidence > 0.7:
                print(f"{confidence:.2f} {pred_name}")
            # else:
                # print(f"{confidence:.2f} uncertain. {pred_name}?")
            s.enter(sample_delay, 1, call_f, (sc,))

        # Set the stream callback to the function above
        stream = sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, blocksize=block_samples)
        stream.start()

        s = sched.scheduler(time.time, time.sleep)
        s.enter(sample_delay, 1, call_f, (s,))
        s.run()

        stream.stop()
        stream.close()
