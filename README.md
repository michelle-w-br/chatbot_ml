# Chatbot intention classification in PyTorch.  

## v1.0
Classify simple intention patterns specified by users, with Bag-of-Words model and a simple layer neural network model
- dataset: dataset/intents.json
- data.py: defines the chat dataset iterator for the dataloader
- bog_encoder: Bag-of-Words (BoW) encoder, computes the vocabulary and input vector for the input data
- model.py: 3-layer nerual network model, 2 hidden layers and output layers
- train.py: train the model based on intention patterns specified in the json file, save the model named as "chat_model.pth"
- main.py: load the trained model "chat_model.pth", test user queries for intention classification

### Usage
Run
```console
python train.py
```
This will save `chat_model.pth` file. And then run
```console
python main.py
```

## v2.0
Intention classification using train/valid/test dataset, with torchtext library (tokenizer) and an embedding model (embedding+fully connected layer)
- dataset/generate_snip_json.py: generate snips_train.json, snips_valid.json, snips_test.json files from SNIP dataset
- data_util.py: defines the chat chataset iterator for the dataloader, tokenizer and vocabulary
- model.py: 2-layer model, token embedding layer and fully connected layer
- train.py: train the model using snips_train.json, evaluate the model using snips_valid.json, and testing the final model performance on snips_test.json 

### Usage
Run
```console
cd dataset
python generate_snip_json.py
cd ..
python train.py
```

## v3.0_kb
creat a customized knowledge base.