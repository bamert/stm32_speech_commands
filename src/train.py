import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# import torchaudio
# from tqdm import tqdm

# from src.dataset import get_train_loader, get_test_loader, SubsetSC
# from src.models import M5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create training and testing split of the data. We do not use validation in this tutorial.


# Resample to 8khz
# transformed = transform(waveform)





# transform = transform.to(device)
# print(model)


# def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)


# n = count_parameters(model)
# print("Number of parameters: %s" % n)



def train(model, transform, train_loader, optimizer, epoch, log_interval):
    model.train()
    for (data, target) in tqdm(train_loader):
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        # if batch_idx % log_interval == 0:
            # print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

