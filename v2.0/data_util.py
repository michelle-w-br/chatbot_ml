from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class ChatDataset(Dataset):
    """
    A single training/test example for simple intent classification.

    Args:
        X: pattern sentence
        Y: intent label
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

def chat_vocab_tokenizer(dataset):
    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data_iter):
            for text,_  in data_iter:
                yield tokenizer(text)    
    vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab, tokenizer