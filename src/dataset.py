import os
import torchaudio.transforms as T
import torch
from torchaudio.datasets import SPEECHCOMMANDS
from pytorch_lightning import LightningDataModule
import torch.utils.data

SC_CLASSES = [
    "background_noise_",
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, transform=None, subset: str = "", new_sample_rate=8000):
        super().__init__("./", download=True)
        self.transform = transform

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]
        self.resample = T.Resample(orig_freq=16000, new_freq=new_sample_rate)
        self.mean = torch.tensor(-2.7432e-06)
        self.std = torch.tensor(0.7073) 
        # Create a dictionary that maps each unique label to a unique integer
        self.label_to_int = {label: i for i, label in enumerate(sorted(SC_CLASSES))}
        self.int_to_label = {i: label for i, label in enumerate(sorted(SC_CLASSES))}

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
        waveform = self.resample(waveform)
        # Pad or trim the waveform to a fixed length (e.g., corresponding to 8000 samples)
        target_length = 8000  # Adjust as needed
        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]  # Trim
        elif waveform.size(1) < target_length:
            # Pad
            padding_size = target_length - waveform.size(1)
            padding = torch.zeros((waveform.size(0), padding_size))
            waveform = torch.cat((waveform, padding), dim=1)

        waveform = (waveform - self.mean) / self.std
        label = torch.tensor(self.label_to_int[label])

        return waveform, label

    @staticmethod
    def num_labels() -> int:
        return len(SC_CLASSES)


class AudioDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, pin_memory):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_set = SubsetSC(subset="training")
        self.val_set = SubsetSC(subset="validation")
        self.test_set = SubsetSC(subset="testing")

    def setup(self, stage=None):
        pass
        #if stage == "fit" or stage is None:
        #    if stage == "test" or stage is None:
    def num_classes(self) -> int:
        return SubsetSC.num_labels()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

