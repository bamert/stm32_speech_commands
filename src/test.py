import torch
import torch.nn.functional as F

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def get_max_activation(tensor):
    return tensor.max()


def evaluate(model, transform, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = transform(data)
            output = model(data)

            # Sum up batch loss
            test_loss += F.nll_loss(output.squeeze(), target, reduction='sum').item()
            # Get the index of the max log-probability
            #pred = output.argmax(dim=1, keepdim=True)
            #correct += pred.eq(target.view_as(pred)).sum().item()
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # Print test results
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

    return test_loss, accuracy


