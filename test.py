# test.py

import torch
import argparse
import pytorch_lightning as pl
#from your_module import AudioClassifier, test_loader  # replace with your actual import
from src.dataset import AudioDataModule
from src.models import AudioClassifier

def main(checkpoint_path):
    model = AudioClassifier.load_from_checkpoint(checkpoint_path)
    data_module = AudioDataModule(batch_size=256, num_workers=4, pin_memory=True, transform=your_transform)

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script for audio classifier.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file to load the model for testing')
    args = parser.parse_args()

    main(args.checkpoint)
