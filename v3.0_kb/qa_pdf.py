import os
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex

os.environ['OPENAI_API_KEY'] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# step 1: upload data and build index
docs = SimpleDirectoryReader('./dataset').load_data()
index = VectorStoreIndex.from_documents(docs)

# step2: setup the query engine
chat_engine = index.as_chat_engine(
    response_mode="compact"
)

# step3: send query
def get_response(msg):
    response = chat_engine.chat(msg)
    return response