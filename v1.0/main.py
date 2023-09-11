import json
import torch

from model import Chat_NeuralNetwork


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
FILE = "chat_model.pth"
data = torch.load(FILE)

model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
intents = data['intents']
tokenizer = data['tokenizer']

model = Chat_NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# testing model using your own query
print("Test your query (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenizer.encode(sentence)
    x = sentence.reshape(1, sentence.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    predicted_intent = intents[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        print(f"intention: {predicted_intent}")
    else:
        print(f"I do not understand...")