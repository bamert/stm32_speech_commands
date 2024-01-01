import torch.nn as nn
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
    def __init__(self, n_input=1, n_output=35, stride=16, kernel_size=80, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
class M4(nn.Module):
    """ 
    Similar to M5, but not part of the paper.
    Uses one layer less due to limited spatial context in M5.conv4 at 8000 input length (1sec 8hkz sample)
    """
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32, use_pool3=True):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.use_pool3 = use_pool3
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        if self.use_pool3:
            x = self.pool3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)



class AudioClassifier(pl.LightningModule):
    def __init__(self, num_labels:int=36):
        super().__init__()
        self.val_correct = 0
        self.val_total = 0
        self.val_pr_correct = 0
        self.val_pr_total = 0
        self.model = M5(n_input=1, n_output=num_labels, kernel_size=10)

    def forward(self, x):
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


        # Calculate accuracy
        probs = torch.exp(log_probs)
        pred = probs.argmax(dim=-1)

        correct = pred.squeeze().eq(target).sum().item()
        self.val_correct += correct
        self.val_total += target.size(0)

        # Calculate pr-acc (post-rejection accuracy)
        # We reject all predictions that have a confidence margin < 0.75
        top2 = probs.topk(2, dim=-1)  # Get top 2 probabilities
        confidence_diff = top2.values[:, 0] - top2.values[:, 1]  
        not_rejected = confidence_diff >= 0.75  # Boolean mask for samples that were not rejected.
        self.val_pr_correct += (pred.squeeze().eq(target) & not_rejected).sum().item()  # Correct and not rejected
        self.val_pr_total += not_rejected.sum().item()  # Total not rejected

        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True)

        # Return a dictionary that includes loss and accuracy
        return {"val_loss": val_loss}
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
    def on_validation_epoch_end(self):

        # Aggregate accuracies
        avg_accuracy = self.val_correct / self.val_total 
        self.log('val_accuracy', avg_accuracy, prog_bar=True, on_epoch=True)


        avg_pr_accuracy = self.val_pr_correct / self.val_pr_total 
        self.log('val_pr_accuracy', avg_pr_accuracy, prog_bar=True, on_epoch=True)
        rejection_percentage = 1. - (self.val_pr_total / self.val_total)
        self.log('val_pct_rejected', rejection_percentage, prog_bar=True, on_epoch=True)
        self.log('val_num_rejected', self.val_total - self.val_pr_total, prog_bar=True, on_epoch=True)


        self.val_total = 0
        self.val_correct = 0
        self.val_pr_total = 0
        self.val_pr_correct = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = {'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1),
                     'interval': 'epoch'}  # Change interval to 'step' if needed
        return [optimizer], [scheduler]

