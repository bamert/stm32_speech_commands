# train.py

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from src.dataset import AudioDataModule
from src.models import AudioClassifier

def main(checkpoint_path=None):
    data_module = AudioDataModule(batch_size=256, num_workers=4, pin_memory=True)
    num_labels = data_module.num_classes()
    model = AudioClassifier(num_labels=num_labels)


    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='audio-classifier-{epoch:02d}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )
    steps_per_epoch = len(data_module.train_dataloader())
    trainer = pl.Trainer(
            #max_epochs=10,  # Adjust as needed
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        val_check_interval=steps_per_epoch  
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for audio classifier.')
    parser.add_argument('--checkpoint', type=str, help='Path to a checkpoint to resume training', default=None)
    args = parser.parse_args()

    main(args.checkpoint)
