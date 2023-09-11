from torch.utils.data import Dataset

class ChatDataset(Dataset):
    """
    A single training/test example for simple intent classification.

    Args:
        X: embedding vector for the sentence
        Y: intent label index
    """
    def __init__(self, X, Y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = Y

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples