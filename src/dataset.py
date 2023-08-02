import os
import torch
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn.functional as F
SC_CLASSES = [
    'background_noise_',
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'down',
    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'go',
    'happy',
    'house',
    'learn',
    'left',
    'marvin',
    'nine',
    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',
    'sheila',
    'six',
    'stop',
    'three',
    'tree',
    'two',
    'up',
    'visual',
    'wow',
    'yes',
    'zero']


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, transform=None, subset: str = ""):
        super().__init__("./", download=True)
        self.transform = transform

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        # Create a mapping of label to index
        # unique_labels = sorted(list(set(self.labels())))
        # self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Initialize an empty set to store unique labels
        # self.unique_labels = set()

        # Iterate over all items in the ParentDataset and add each unique label to the set
        # for i in range(len(self)):
            # item = super().__getitem__(i)
            # self.unique_labels.add(item[2])
        # Create a dictionary that maps each unique label to a unique integer
        self.label_to_int = {label: i for i, label in enumerate(sorted(SC_CLASSES))}
        self.int_to_label= {i: label for i, label in enumerate(sorted(SC_CLASSES))}

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
    def __getitem__(self, index):
        item = super().__getitem__(index)
        waveform = item[0]
        label = item[2]
        # Map the label to an integer
        label = torch.tensor(self.label_to_int[label]) 

        # One-hot encode  NOTE: not needed
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

    def num_labels(self) -> int:
        return len(self.label_to_int.keys())
def get_train_loader(train_set, batch_size: int, collate_fn, num_workers, pin_memory):
    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
def get_test_loader(test_set, batch_size: int, collate_fn, num_workers, pin_memory):
    return  torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
