import torch


def label_to_index(labels, word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


# def index_to_label(index):
# # Return the word corresponding to the index in labels
# # This is the inverse of label_to_index
# return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)
