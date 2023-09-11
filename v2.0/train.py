import json
import numpy as np
import torch
import torch.nn as nn
import time

from data_util import ChatDataset, chat_vocab_tokenizer
from model import TextClassificationModel

from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset


# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
total_accu = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(file_name):
    # load json data
    with open(file_name, 'r') as f:
        doc=json.load(f)

    labels=[]
    sentences=[]
    sentences_label=[]
    sentences_label_idx=[]
    for intent in doc.keys():
        labels.append(intent)
        for pattern in doc[intent]['patterns']:
            sentences.append(pattern)
            sentences_label.append(intent)

    for label in sentences_label:
        label_idx=labels.index(label)
        sentences_label_idx.append(label_idx)

    return sentences,sentences_label_idx,labels

train_x,train_y,labels = load_json("./dataset/snips_train.json")
valid_x,valid_y,_ = load_json("./dataset/snips_valid.json")
test_x,test_y,_ = load_json("./dataset/snips_test.json")
print(len(train_x))
print(len(train_y))

train_iter = ChatDataset(train_x,train_y)
valid_iter = ChatDataset(valid_x,valid_y)
test_iter = ChatDataset(test_x,test_y)

#tokenzier and vacoab
vocab,tokenizer = chat_vocab_tokenizer(train_iter)
print(vocab(['here', 'is', 'an', 'example']))

#dataloader
text_pipeline = lambda x: vocab(tokenizer(x))

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list.to(device), label_list.to(device), offsets.to(device)

train_dataset = to_map_style_dataset(train_iter)
valid_dataset = to_map_style_dataset(valid_iter)
test_dataset = to_map_style_dataset(test_iter)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

#define model
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

#define train and evalue
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 50
    start_time = time.time()

    for idx, (text, label, offsets) in enumerate(dataloader):
        #print(idx)
        #print(label, text, offsets)
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)


print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))