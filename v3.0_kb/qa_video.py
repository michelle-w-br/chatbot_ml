import os
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from llama_index import VectorStoreIndex

os.environ['OPENAI_API_KEY'] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

loader = YoutubeTranscriptReader()
documents = loader.load_data(ytlinks=['https://youtu.be/1ZwXkw9_Xq8'])

index = VectorStoreIndex.from_documents(documents)

# step2: setup the query engine
chat_engine = index.as_chat_engine(
    response_mode="compact"
)

# step3: send query
def get_response_videoTransp(msg):
    response = chat_engine.chat(msg)
    return response