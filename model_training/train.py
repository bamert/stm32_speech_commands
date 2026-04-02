import argparse
import onnx
import itertools
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from dataset import AudioDataModule
from models import AudioClassifier

from torchinfo import summary
import numpy as np
import torch
from dataset import AudioDataModule
from models import AudioClassifier

def get_param_grid(mode):
    """Yields dictionaries of hyperparameter combinations for grid search."""
    if mode == "1d":
        channels = [16, 32]
        blocks = [3, 4]
        front_doors = [(10, 5), (16, 16), (80, 16)]  # (kernel_size, stride)
        pools = [4]
        
        for c, b, (k, s), p in itertools.product(channels, blocks, front_doors, pools):
            yield {
                "n_channel": c, 
                "n_blocks": b, 
                "kernel_size": k, 
                "stride": s, 
                "pool_size": p
            }
            
    elif mode == "2d":
        channels = [32, 64]
        blocks = [1, 2]
        input_res = [(20, 256), (40, 128)]  # (n_mels, hop_length)
        
        for c, b, (m, h) in itertools.product(channels, blocks, input_res):
            yield {
                "n_channel": c, 
                "n_blocks": b, 
                "n_mels": m, 
                "hop_length": h
            }
    else:
        raise ValueError("Mode must be '1d' or '2d'")

def get_accelerator():
    if torch.cuda.is_available(): return 'gpu' 
    elif torch.backends.mps.is_available(): return 'mps'
    else: return 'cpu'
def fix_onnx(onnx_path):
    "Removes onnx nodes that stedgeai cannot process. File is updated in-place."
    onnx_model = onnx.load(onnx_path)
    fixed_count = 0

    for node in onnx_model.graph.node:
        if node.op_type == "Reshape":
            cleaned_attributes = [a for a in node.attribute if a.name != 'allowzero']
            if len(node.attribute) != len(cleaned_attributes):
                del node.attribute[:]
                node.attribute.extend(cleaned_attributes)
                fixed_count += 1

    onnx.save(onnx_model, onnx_path)
def run_experiment(params: dict, mode: str, sample_rate_hz: int, checkpoint_suffix:str, checkpoint_path=None):
    """Runs a single training session and returns the best accuracy and edge profile."""
    print(f"\n--- Starting Experiment: {mode.upper()} | Params: {params} ---")
    
    data_module = AudioDataModule(
        batch_size=256, num_workers=4, pin_memory=True, 
        sample_rate_hz=sample_rate_hz, mode=mode
    )
    data_module.prepare_data()
    data_module.setup()
    num_labels = data_module.num_classes()
    
    classifier = AudioClassifier(num_labels=num_labels, mode=mode, **params)
    
    if mode == "1d":
        dummy_shape = (1, 1, sample_rate_hz)
    else:
        mels = params.get("n_mels", 40)
        hop = params.get("hop_length", 128)
        frames = (sample_rate_hz // hop) + 1
        dummy_shape = (1, 1, mels, frames)
        
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        # Add a identifier if running a grid so checkpoints don't overwrite
        filename=f'audio-classifier-{mode}-' + '{val_accuracy:.3f}-{epoch:02d}',
        save_top_k=1,
        monitor='val_accuracy',
        mode='max',
    )
    
    steps_per_epoch = len(data_module.train_dataloader())
    trainer = pl.Trainer(
        max_epochs=35,  
        accelerator=get_accelerator(),
        devices=1,
        callbacks=[checkpoint_callback],
        val_check_interval=steps_per_epoch,
        enable_model_summary=False # Turn off default PL summary to avoid console spam
    )

    trainer.fit(classifier, datamodule=data_module, ckpt_path=checkpoint_path)
    best_model = AudioClassifier.load_from_checkpoint(checkpoint_callback.best_model_path).model
    best_model.eval()
    
    onnx_path = checkpoint_callback.best_model_path.replace('.ckpt', f'_{checkpoint_suffix}.onnx')
    device = next(best_model.parameters()).device
    dummy_input = torch.randn(dummy_shape, device=device)
    
    print(f"Exporting best model to ONNX: {onnx_path}")
    # PyTorch defaults to a single file unless the model is >2GB
    torch.onnx.export(
        best_model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=8,       
        input_names=['input'],
        output_names=['output']
    ) 
    fix_onnx(onnx_path)
    best_acc = checkpoint_callback.best_model_score.item()
    
    return best_acc 

def main(args):
    sample_rate_hz = 8000
    
    if args.grid:
        print(f"Grid Search for {args.mode.upper()}...")
        grid = get_param_grid(args.mode)
        results_file = f"grid_results_{args.mode}.txt"
        
        # Create/overwrite the file and write the header
        with open(results_file, "w") as f:
            f.write("ID | Accuracy | MACs (M) | Params | Peak SRAM | Hyperparameters\n")
            f.write("-" * 80 + "\n")
            
        for i, params in enumerate(grid):
            # Run the training
            best_acc = run_experiment(params, args.mode, sample_rate_hz, i)
            
            # Append result to text file immediately so you don't lose data if it crashes!
            with open(results_file, "a") as f:
                res_line = f" {i} | {best_acc:.4f} | {prof['MACs (Millions)']}M | {prof['Total Parameters']} | {prof['Peak Activation (Elements)']} | {params}\n"
                f.write(res_line)
                
        print(f" Grid search complete! Results saved to {results_file}")
        
    else:
        best_acc, prof = run_experiment({}, args.mode, sample_rate_hz, "", args.checkpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for audio classifier.')
    parser.add_argument('--checkpoint', type=str, help='Path to a checkpoint to resume training', default=None)
    parser.add_argument('--mode', type=str, choices=['1d', '2d'], default='1d', help='Choose pipeline to run')
    parser.add_argument('--grid', action='store_true', help='Flag to run the hyperparameter grid search')
    args = parser.parse_args()

    main(args)

