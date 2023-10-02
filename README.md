# Custom Chatbot in Python
- medium: https://medium.com/@bridgeriver-ai
- youtube: https://www.youtube.com/@bridgeriver_ai

## v1.0: botpress/voiceflow intent with few data
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

### Watch the Tutorial
[![Alt text](http://i3.ytimg.com/vi/S-pse__T3qE/hqdefault.jpg)](https://youtu.be/S-pse__T3qE)  
[https://youtu.be/S-pse__T3qE](https://youtu.be/S-pse__T3qE)


## v2.0: botpress/voiceflow intent with more data
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


## v3.0_kb: : botpress/voiceflow knowledge base
creat a custom chatbot with your data, similar to knowledge base feature in Botpress/Voiceflow.
- dataset/llama2.pdf: user data, can be any format supported by llama hub in llama-index
- chatpdf.py: main file to start the chatbot server (query pdf)
- qa_pdf.py: set up llama index for user data (pdf data)
- chatvideo.py: main file to start the chatbot server (query youtube video transcript)
- qa_video: set up llama index for user data (youtube video transcript data)
- templates/home.html: home landing page for the server
- static/app.js, static/style.css: front end code

### Usage
Run
```console
pip install llama_index
pip install openai
python chatpdf.py

pip install llama_hub
pip install youtube_transcript_api
python chat_video
```

### frontend code credits:
Frontend code credits go to:
https://github.com/hitchcliff/front-end-chatjs

### Watch the Tutorial
[![Alt text](http://i3.ytimg.com/vi/cnpn_HDazY4/hqdefault.jpg)](https://youtu.be/cnpn_HDazY4)  
[https://youtu.be/cnpn_HDazY4](https://youtu.be/cnpn_HDazY4)
