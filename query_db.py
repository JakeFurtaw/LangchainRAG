"""
This script is used to query the database of text chunks created in create_db.py.
It uses the langchain library to load the model and tokenizer. 
Then the script queries the database and returns the result or results depending on the k #.
"""
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

#load the huggingface api token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
login(token = HUGGINGFACE_API_TOKEN)

#load the llama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = HUGGINGFACE_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token = HUGGINGFACE_API_TOKEN)

CHROMA_PATH = 'chroma'
#chat template to get better results from llama model
LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are a helpful, respectful and honest AI assistant to help college students navigate a college campus."
    "Always answer as helpfully as possible, while being safe."
    "Please ensure that your responses are clear, concise, and positive in nature."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
)
def main():
    #load the database
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #query the database
    if len(sys.argv) > 1:
        #get the query
        query=' '.join(sys.argv[1:])
        #query the db for the most similar results
        results = db.similarity_search_with_relevance_scores(query, k=2)
        if len (results) == 0:
            print("No results found.")
            return
        else:
            #printing query(Testing purposes only)
            print("-------------------------------------------------------")
            print (f"\nQuery: " + query)
            #print the results
            print("\nResults:")
            for result in results:
                print(result)     
    else:
        print("Please provide a query.")

if __name__ == '__main__':
    main()


