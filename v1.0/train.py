import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from bog_encoder import BOW_Encoder
from data import ChatDataset
from model import Chat_NeuralNetwork


# load json data
print("----------- step 1: load json dataset...")
with open('./dataset/intents.json', 'r') as f:
    doc=json.load(f)

intents=[]
sentences=[]
train_data=[]
y=[]
for intent in doc.keys():
    intents.append(intent)
    for pattern in doc[intent]['patterns']:
        sentences.append(pattern)
        y.append(intent)
        train_data.append((pattern,intent))

print("intent size is: ", len(intents))
print("intent include: ", intents)
print("sentences size is: ", len(sentences))
print("train dataset size is: ", len(train_data))


# generate numpy dataset
print("----------- step 2: preparing numpy dataset...")
tokenizer=BOW_Encoder(sentences)
x_train=[]
y_train=[]
for (x,y) in train_data:
    embedding=tokenizer.encode(x)
    x_train.append(embedding)
    label=intents.index(y)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


#generate pytorch dataset
print("----------- step 3: preparing pytorch dataset...")
chat_dataset = ChatDataset(x_train,y_train)

#load dataset
print("----------- step 4: lodading dataset...")
chat_dataset_loader=DataLoader(dataset=chat_dataset,
                        batch_size=8,
                        shuffle=True,
                        num_workers=0)


# model architecture definition
print("----------- step 5: preparing model...")
input_size=len(x_train[0])
hidden_size=8
output_size=len(intents)
num_epochs=1000
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Chat_NeuralNetwork(input_size, hidden_size, output_size).to(device)


# Train the model
print("----------- step 6: training the model...")
# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (patterns, labels) in chat_dataset_loader:
        patterns = patterns.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(patterns)
        loss = loss_func(outputs, labels) #labels should be class indices, not one-hot vector
        
        # Backward and optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    if epoch % 100 == 0:
        print (f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')
print(f'final loss: {loss.item()}')


# save the model
print("step 7: save the model...")
chat_model = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"intents": intents,
"tokenizer":tokenizer
}

FILE = "chat_model.pth"
torch.save(chat_model, FILE)

print(f'training complete. file saved to {FILE}')