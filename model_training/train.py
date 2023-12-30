import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from dataset import AudioDataModule
from models import AudioClassifier
def export_onnx_model(model, audio_dim, path):
    # Input to the model
    x = torch.randn(1, 1, audio_dim, requires_grad=True)

    # Export the model
    torch.onnx.export(model,               
                  x,                         
                  path,   
                  export_params=True,        
                  opset_version=7,          
                  do_constant_folding=True,  
                  input_names = ['input'],  
                  output_names = ['output'], 
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})

def main(checkpoint_path=None, sample_rate_hz=8000):
    data_module = AudioDataModule(batch_size=256, num_workers=4, pin_memory=True, sample_rate_hz=sample_rate_hz)
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
            max_epochs=12,  # Adjust as needed
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        val_check_interval=steps_per_epoch  
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
    export_onnx_model(model, sample_rate_hz, f"model.onnx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for audio classifier.')
    parser.add_argument('--checkpoint', type=str, help='Path to a checkpoint to resume training', default=None)
    args = parser.parse_args()

    main(args.checkpoint)
