import torch
import torch.optim as optim
import torchaudio
from tqdm import tqdm

import torch.nn.functional as F
from src.dataset import get_train_loader, get_test_loader, SubsetSC
from src.models import M5
from src.train import train_step
from src.test import evaluate
from src.collate import collate_fn


def export_onnx_model(model, audio_dim, path):
    # Input to the model
    x = torch.randn(1, 1, audio_dim, requires_grad=True)
    # torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=7,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

new_sample_rate = 8000
batch_size = 256
transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)
train_set = SubsetSC(subset="training")
test_set = SubsetSC(subset="testing")


train_loader = get_train_loader(
    train_set, batch_size, collate_fn, num_workers, pin_memory
)
test_loader = get_test_loader(test_set, batch_size, collate_fn, num_workers, pin_memory)

model = M5(n_input=1, n_output=train_set.num_labels())

model.to(device)
transform.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.1
)  # reduce the learning after 5 epochs by a factor of 10
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train_and_evaluate(model, transform, train_loader, test_loader, optimizer, scheduler, total_steps, eval_interval, start_step=0):
    model.train()
    step = start_step
    train_iter = iter(train_loader)

    pbar = tqdm(total=total_steps, initial=start_step, desc="Training", dynamic_ncols=True)

    while step < total_steps:
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)

        loss = train_step(model, transform, data, target, optimizer)
        #scheduler.step()

        pbar.update(1)
        #pbar.set_postfix(loss=loss, step=step, next_eval=step % eval_interval)
        next_eval_countdown = eval_interval - (step % eval_interval)
        pbar.set_postfix(loss=loss, step=step, next_eval=next_eval_countdown)

        if step % eval_interval == 0 and step != 0:
            model.eval()  # Switch to evaluation mode
            evaluate(model, transform, test_loader, device)  # Your evaluation function
            model.train()  # Switch back to training mode

        step += 1

    pbar.close()

# Example usage
train_and_evaluate(model, transform, train_loader, test_loader, optimizer, scheduler, total_steps=10000, eval_interval=500)

if __name__ == "__main__":
    n_epoch = 20
    log_interval = 20
    total_steps = len(train_loader) * n_epoch
    eval_interval = int(0.5*len(train_loader)) # Evaluate once every epoch
    start_step = 0
    train_and_evaluate(model, transform, train_loader, test_loader, optimizer, scheduler, total_steps, eval_interval, start_step)
    #torch.save(model.state_dict(), "model")
    #export_onnx_model(model, new_sample_rate, "model_train.onnx")
