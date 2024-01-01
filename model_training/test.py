import torch
import argparse
import pytorch_lightning as pl
from dataset import AudioDataModule
from models import AudioClassifier

def main(checkpoint_path):
    data_module = AudioDataModule(batch_size=256, num_workers=4, pin_memory=True, sample_rate_hz=8000)
    num_labels = data_module.num_classes()
    model = AudioClassifier.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1)
    trainer.validate(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script for audio classifier.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file to load the model for testing')
    args = parser.parse_args()

    main(args.checkpoint)
