import torch
from .util import label_to_index, pad_sequence
def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    # mean = torch.tensor(-3.3276e-07)
    # std = torch.tensor(0.1216)
    mean = torch.tensor(-2.7432e-06)
    std= torch.tensor(0.7073)
    for waveform, label, in batch:
        waveform = (waveform - mean) / std
        tensors += [waveform]
        # Label is already an integer
        targets += [label] # label_to_index(labels, label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


