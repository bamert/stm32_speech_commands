import torch

from src.dataset import get_train_loader, SubsetSC
from src.collate import collate_fn

# Assuming that dataset is your instance of a Dataset subclass


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    batch_size = 256
    sample_length = 16000
    train_set = SubsetSC(subset="training")
    loader = get_train_loader(
        train_set, batch_size, collate_fn, num_workers, pin_memory
    )
    mean = 0.0
    variance = 0.0
    for samples, _ in loader:
        mean += samples.mean()
    mean = mean / len(loader.dataset)

    for samples, _ in loader:
        variance += ((samples - mean) ** 2).sum()
    std = torch.sqrt(variance / (len(loader.dataset) * sample_length))
    print("mean: ", mean)
    print("std: ", std)
