import os
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex

os.environ['OPENAI_API_KEY'] = "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"

# step 1: upload data and build index
docs = SimpleDirectoryReader('./dataset').load_data()
index = VectorStoreIndex.from_documents(docs)

# step2: setup the query engine
query_engine = index.as_query_engine(
    response_mode="compact"
)

# step3: send query
def get_response(msg):
    response = query_engine.query(msg)    
    return response
