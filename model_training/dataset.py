import scipy.io.wavfile as wavfile
import torch
import os
import torchaudio.transforms as T
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
    def __init__(
        self, transform=None, subset: str = "", new_sample_rate=8000, mode="1d"
    ):
        super().__init__("./", download=True)
        self.transform = transform
        self.mode = mode
        self.new_sample_rate = new_sample_rate

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]

        self.resample = T.Resample(orig_freq=16000, new_freq=self.new_sample_rate)
        self.mean = torch.tensor(-2.7432e-06)
        self.std = torch.tensor(0.7073)

        if self.mode == "2d":
            self.mel_spec = T.MelSpectrogram(
                sample_rate=self.new_sample_rate,
                n_fft=256,
                win_length=256,
                hop_length=128,
                n_mels=40,
            )
            self.db_transform = T.AmplitudeToDB(top_db=80)

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
        file_path = self._walker[index]

        # Extract the label directly from the folder name
        label = os.path.basename(os.path.dirname(file_path))

        # We avoid torchaudio here due to incompatibility with ARM linux (GB10 devices)
        sample_rate, waveform_np = wavfile.read(file_path)
        # Convert Numpy array to PyTorch tensor [1, Time] and normalize
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0) / 32768.0

        # Resample if necessary
        if sample_rate != self.new_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.new_sample_rate)
            waveform = resampler(waveform)

        # Pad or trim to exactly 8000 samples
        if waveform.size(1) > self.new_sample_rate:
            waveform = waveform[:, : self.new_sample_rate]
        elif waveform.size(1) < self.new_sample_rate:
            padding_size = self.new_sample_rate - waveform.size(1)
            padding = torch.zeros((waveform.size(0), padding_size))
            waveform = torch.cat((waveform, padding), dim=1)

        label_int = torch.tensor(self.label_to_int[label])
        return waveform, label_int

    @staticmethod
    def num_labels() -> int:
        return len(SC_CLASSES)


class AudioDataModule(LightningDataModule):
    def __init__(
        self, batch_size, num_workers, pin_memory, sample_rate_hz: int = 8000, mode="1d"
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.new_sample_rate = sample_rate_hz
        self.mode = mode
        self.train_set = SubsetSC(
            subset="training", new_sample_rate=self.new_sample_rate, mode=self.mode
        )
        self.val_set = SubsetSC(
            subset="validation", new_sample_rate=self.new_sample_rate, mode=self.mode
        )
        self.test_set = SubsetSC(
            subset="testing", new_sample_rate=self.new_sample_rate, mode=self.mode
        )

    def setup(self, stage=None):
        pass

    @staticmethod
    def num_classes() -> int:
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
