"""
This script is used to query the database of text chunks created in create_db.py.
It uses the langchain library to load the database, query the database, and return the most similar text chunks.
"""
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import ChatPromptTemplate
import os

#load the huggingface api token
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
login(token = HUGGINGFACE_API_TOKEN)

#load the llama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

CHROMA_PATH = 'chroma'
#chat template to get better results from llama model
LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are a helpful, respectful and honest AI assistant to help college student navigate a college campus."
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
    query = "\n\nWhat is Jal Irani Room Number?"
    results = db.similarity_search_with_relevance_scores(query, k=3)
    if len (results) == 0:
        print("No results found.")
        return
    
    #loading prompt template/formatting the prompt/printing it
    #prompt_template = ChatPromptTemplate(LLAMA_CHAT_TEMPLATE)
    prompt = LLAMA_CHAT_TEMPLATE.format(query_str=query, context_str=results[0])
    print (query)

    #print the results
    print("\nResults:")
    for result in results:
        print(result)
       
if __name__ == '__main__':
    main()


