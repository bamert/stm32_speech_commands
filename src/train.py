import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, transform, data, target, optimizer):
    data, target = data.to(device), target.to(device)
    
    # Apply transform and model on the whole batch directly on the device
    data = transform(data)
    output = model(data)

    # Calculate loss (e.g., negative log-likelihood)
    loss = F.nll_loss(output.squeeze(), target)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


