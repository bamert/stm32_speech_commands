import torch.nn as nn
import torchaudio.transforms as T
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import torch.optim as optim


class M5(nn.Module):
    """
    Model from the following paper
    "Very deep convolutional neural networks for raw waveforms,"
    W. Dai, C. Dai, S. Qu, J. Li and S. Das,
    2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
    New Orleans, LA, USA, 2017, pp. 421-425, doi: 10.1109/ICASSP.2017.7952190.
    """

    def __init__(
        self,
        n_input=1,
        n_output=35,
        stride=16,
        kernel_size=64,
        n_channel=32,
        n_blocks=4,
        pool_size=4,
    ):
        super().__init__()
        input_layer = []
        input_layer.append(
            nn.Conv1d(n_input, n_channel, kernel_size=kernel_size, stride=stride)
        )
        input_layer.append(nn.BatchNorm1d(n_channel))
        self.input_features = nn.Sequential(*input_layer)
        self.pool = nn.MaxPool1d(pool_size)

        in_channels = n_channel

        layers = []
        for i in range(1, n_blocks):
            out_channels = n_channel * (2 ** (i // 2))

            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, n_output)

    def forward(self, x):
        x = self.input_features(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.features(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class Dilated1DNano(nn.Module):
    def __init__(self, n_output=36, base_channels=24):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, base_channels, kernel_size=24, stride=16, padding=12, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(4, stride=4)

        self.dw1 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            groups=base_channels,
            bias=False,
        )
        self.pw1 = nn.Conv1d(base_channels, 32, kernel_size=1, bias=False)
        self.bn1_dw = nn.BatchNorm1d(32)

        self.dw2 = nn.Conv1d(
            32, 32, kernel_size=3, padding=2, dilation=2, groups=32, bias=False
        )
        self.pw2 = nn.Conv1d(32, 36, kernel_size=1, bias=False)
        self.bn2_dw = nn.BatchNorm1d(36)

        self.dw3 = nn.Conv1d(
            36, 36, kernel_size=3, padding=4, dilation=4, groups=36, bias=False
        )
        self.pw3 = nn.Conv1d(36, 36, kernel_size=1, bias=False)
        self.bn3_dw = nn.BatchNorm1d(36)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(36, n_output)

    def forward(self, x):
        # Log Compression (emulate Mel)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.abs(x)
        x = self.pool1(x)
        x = torch.log(x + 1e-5)

        x = F.relu(self.pw1(self.dw1(x)))
        x = self.bn1_dw(x)

        x = F.relu(self.pw2(self.dw2(x)))
        x = self.bn2_dw(x)

        x = F.relu(self.pw3(self.dw3(x)))
        x = self.bn3_dw(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Dilated1DMicro(nn.Module):
    def __init__(self, n_output=36, base_channels=24):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, base_channels, kernel_size=64, stride=32, padding=16, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(2, stride=2)

        self.dw1 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            groups=base_channels,
            bias=False,
        )
        self.pw1 = nn.Conv1d(base_channels, base_channels, kernel_size=1, bias=False)
        self.bn1_dw = nn.BatchNorm1d(base_channels)

        self.dw2 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=base_channels,
            bias=False,
        )
        self.pw2 = nn.Conv1d(base_channels, 32, kernel_size=1, bias=False)
        self.bn2_dw = nn.BatchNorm1d(32)

        self.dw3 = nn.Conv1d(
            32, 32, kernel_size=3, padding=4, dilation=4, groups=32, bias=False
        )
        self.pw3 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.bn3_dw = nn.BatchNorm1d(32)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, n_output)

    def forward(self, x):
        # Log Compression (emulate Mel)
        x = self.conv1(x)
        x = self.bn1(x)

        x = torch.abs(x)
        x = self.pool1(x)
        x = torch.log(x + 1e-5)

        x = F.relu(self.pw1(self.dw1(x)))
        x = self.bn1_dw(x)

        x = F.relu(self.pw2(self.dw2(x)))
        x = self.bn2_dw(x)

        x = F.relu(self.pw3(self.dw3(x)))
        x = self.bn3_dw(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Dilated1DBalanced(nn.Module):
    def __init__(self, n_output=36, base_channels=24):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, base_channels, kernel_size=32, stride=16, padding=32, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(4, stride=4)

        self.dw1 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            groups=base_channels,
            bias=False,
        )
        self.pw1 = nn.Conv1d(base_channels, 32, kernel_size=1, bias=False)
        self.bn1_dw = nn.BatchNorm1d(32)

        self.dw2 = nn.Conv1d(
            32, 32, kernel_size=3, padding=2, dilation=2, groups=32, bias=False
        )
        self.pw2 = nn.Conv1d(32, 48, kernel_size=1, bias=False)
        self.bn2_dw = nn.BatchNorm1d(48)

        self.dw3 = nn.Conv1d(
            48, 48, kernel_size=3, padding=4, dilation=4, groups=48, bias=False
        )
        self.pw3 = nn.Conv1d(48, 48, kernel_size=1, bias=False)
        self.bn3_dw = nn.BatchNorm1d(48)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, n_output)

    def forward(self, x):
        # Log Compression (emulate Mel)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.abs(x)
        x = self.pool1(x)
        x = torch.log(x + 1e-5)

        x = F.relu(self.pw1(self.dw1(x)))
        x = self.bn1_dw(x)

        x = F.relu(self.pw2(self.dw2(x)))
        x = self.bn2_dw(x)

        x = F.relu(self.pw3(self.dw3(x)))
        x = self.bn3_dw(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Dilated1D(nn.Module):
    def __init__(self, n_output=36, base_channels=32):
        super().__init__()

        self.conv1 = nn.Conv1d(
            1, base_channels, kernel_size=128, stride=16, padding=64, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool1 = nn.MaxPool1d(4, stride=4)

        self.dw1 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            groups=base_channels,
            bias=False,
        )
        self.pw1 = nn.Conv1d(base_channels, base_channels, kernel_size=1, bias=False)
        self.bn1_dw = nn.BatchNorm1d(base_channels)

        self.dw2 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=base_channels,
            bias=False,
        )
        self.pw2 = nn.Conv1d(
            base_channels, base_channels * 2, kernel_size=1, bias=False
        )
        self.bn2_dw = nn.BatchNorm1d(base_channels * 2)

        self.dw3 = nn.Conv1d(
            base_channels * 2,
            base_channels * 2,
            kernel_size=3,
            padding=4,
            dilation=4,
            groups=base_channels * 2,
            bias=False,
        )
        self.pw3 = nn.Conv1d(
            base_channels * 2, base_channels * 2, kernel_size=1, bias=False
        )
        self.bn3_dw = nn.BatchNorm1d(base_channels * 2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels * 2, n_output)

    def forward(self, x):
        # Log Compression (Fake Mel)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.abs(x)
        x = self.pool1(x)
        x = torch.log(x + 1e-5)

        x = F.relu(self.pw1(self.dw1(x)))
        x = self.bn1_dw(x)

        x = F.relu(self.pw2(self.dw2(x)))
        x = self.bn2_dw(x)

        x = F.relu(self.pw3(self.dw3(x)))
        x = self.bn3_dw(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class DSCNN(nn.Module):
    def __init__(self, n_input=1, n_output=36, n_channel=64, n_blocks=2):
        super().__init__()
        self.n_blocks = n_blocks

        self.conv1 = nn.Conv2d(
            n_input, n_channel, kernel_size=(5, 4), stride=(2, 2), padding=(2, 1)
        )
        self.bn1 = nn.BatchNorm2d(n_channel)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dw1 = nn.Conv2d(
            n_channel, n_channel, kernel_size=3, padding=1, groups=n_channel, bias=False
        )
        self.bn_dw1 = nn.BatchNorm2d(n_channel)
        self.pw1 = nn.Conv2d(n_channel, n_channel, kernel_size=1, bias=False)
        self.bn_pw1 = nn.BatchNorm2d(n_channel)

        self.dw2 = nn.Conv2d(
            n_channel, n_channel, kernel_size=3, padding=1, groups=n_channel, bias=False
        )
        self.bn_dw2 = nn.BatchNorm2d(n_channel)
        self.pw2 = nn.Conv2d(n_channel, n_channel, kernel_size=1, bias=False)
        self.bn_pw2 = nn.BatchNorm2d(n_channel)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn_dw1(self.dw1(x)))
        x = F.relu(self.bn_pw1(self.pw1(x)))

        if self.n_blocks == 2:
            x = F.relu(self.bn_dw2(self.dw2(x)))
            x = F.relu(self.bn_pw2(self.pw2(x)))

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class AudioClassifier(pl.LightningModule):
    def __init__(self, num_labels: int = 36, mode="1d", **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.val_correct = 0
        self.val_total = 0
        self.val_pr_correct = 0
        self.val_pr_total = 0
        self.mode = mode

        if self.mode == "1d":
            # self.model = M5(n_input=1, n_output=num_labels, log_compress=True, **kwargs)
            self.model = Dilated1DNano(n_output=num_labels, base_channels=16)

            self.register_buffer("mean", torch.tensor(-2.7432e-06))
            self.register_buffer("std", torch.tensor(0.7073))

        elif self.mode == "2d":
            # DSP kwargs
            mels = kwargs.pop("n_mels", 20)
            hop = kwargs.pop("hop_length", 256)

            self.model = DSCNN(n_input=1, n_output=num_labels, **kwargs)

            self.mel_spec = T.MelSpectrogram(
                sample_rate=8000, n_fft=256, win_length=256, hop_length=hop, n_mels=mels
            )
            self.db_transform = T.AmplitudeToDB(top_db=80)

    def forward(self, x):
        if self.mode == "1d":
            x = (x - self.mean) / self.std
        elif self.mode == "2d":
            mel = self.mel_spec(x)
            log_mel = self.db_transform(mel)
            x = (log_mel + 40.0) / 40.0
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output.squeeze(), target)
        return loss

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        data, target = batch
        log_probs = self(data)
        val_loss = F.nll_loss(log_probs.squeeze(), target)

        probs = torch.exp(log_probs)
        pred = probs.argmax(dim=-1)

        correct = pred.squeeze().eq(target).sum().item()
        self.val_correct += correct
        self.val_total += target.size(0)

        # Calculate pr-acc (post-rejection accuracy)
        # We reject all predictions that have a confidence margin < 0.75
        top2 = probs.topk(2, dim=-1)  # Get top 2 probabilities
        confidence_diff = top2.values[:, 0] - top2.values[:, 1]
        not_rejected = (
            confidence_diff >= 0.75
        )  # Boolean mask for samples that were not rejected.
        self.val_pr_correct += (
            (pred.squeeze().eq(target) & not_rejected).sum().item()
        )  # Correct and not rejected
        self.val_pr_total += not_rejected.sum().item()  # Total not rejected

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)

        return {"val_loss": val_loss}

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def on_validation_epoch_end(self):

        # Aggregate accuracies
        avg_accuracy = self.val_correct / self.val_total if self.val_total > 0 else 0
        self.log("val_accuracy", avg_accuracy, prog_bar=True, on_epoch=True)

        avg_pr_accuracy = (
            self.val_pr_correct / self.val_pr_total if self.val_pr_total > 0 else 0
        )
        self.log("val_pr_accuracy", avg_pr_accuracy, prog_bar=True, on_epoch=True)
        rejection_percentage = 1.0 - (self.val_pr_total / self.val_total)
        self.log("val_pct_rejected", rejection_percentage, prog_bar=True, on_epoch=True)
        self.log(
            "val_num_rejected",
            self.val_total - self.val_pr_total,
            prog_bar=True,
            on_epoch=True,
        )

        self.val_total = 0
        self.val_correct = 0
        self.val_pr_total = 0
        self.val_pr_correct = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = {
            "scheduler": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
