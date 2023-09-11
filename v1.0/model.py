import torch.nn as nn


class Chat_NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Chat_NeuralNetwork, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.hidden_layer1(x)
        out = self.relu(out)
        out = self.hidden_layer2(out)
        out = self.relu(out)
        out = self.output(out)
        
        return out