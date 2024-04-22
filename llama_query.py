"""
This script is used to query the database of text chunks created in create_db.py.
It uses the langchain library to load the model and tokenizer.
Then the script queries the database and returns the result or results depending on the k #.
MAKE SURE TO REIGNORE THE .env FILE AFTER USE
"""
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from textwrap import wrap
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

# Path to the Chroma database and Embedding Model
CHROMA_PATH = 'TowsonDB'
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# Chat template to get better results from the model
CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are an AI Assistant that helps college students navigate a college campus."
    "Dont answer any questions about or give any information about any other school/university besides Towson University."
    "You provide information like teacher and faculty contact information, teachers office room numbers, course information, enrollment information, campus resources," 
    "and general campus information."
    "Please ensure that your responses are clear, concise, and positive in nature."
    "If you dont know the answer to a question, you can say that you are not sure."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query} [/INST]"
)
# Specify the GPU as device if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Load your Hugging Face API token
load_dotenv(Path(".env"))
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
#Configure the quantization config
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# Load the Model and Tokenizer    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B", 
    quantization_config=quantization_config,
    chat_template=CHAT_TEMPLATE,
    device_map="auto")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B", 
    quantization_config=quantization_config,
    device_map="auto")
# Printing Results to the CLI
def print_results(results):
    if not results:
        print("Sorry, I couldn't find any relevant information for your query.")
        return
    print("\nResults:")
    print('-' * 80)
    response_text = results[0]
    print("\n"+response_text)
    print('-' * 80)
# Main Function
def main():
    # Load the database and the embedding function
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, 
                embedding_function=embeddings)
    while True:
        # Query the database
        query = input("Enter query: ")
        if len(query) == 0:
            print("Please enter a query.")
            continue
        elif query.lower() == "exit":
                break
        # Query the db for the most similar results
        search_results=db.similarity_search_with_relevance_scores(query, k=5)
        #combine search results from db search to form context string
        docs = []
        for result in search_results:
            document, score = result
            docs.append(document.page_content.strip())
        context_str = "\n".join(docs)
        print(context_str)
        # Move the input tensors to the device
        input_tensors = tokenizer(query,
                                return_tensors="pt").to(device)
        # Generate the response from the model
        response = model.generate(**input_tensors, 
                                    max_new_tokens=512,
                                    temperature= .1,
                                    do_sample=True)
        response_text = tokenizer.decode(response[0], 
                                        skip_special_tokens=True)
        # Print the results and query
        print('-' * 80)
        print(f"Query: {query}")
        print_results([(f"\n"+response_text)])
        print(f"Relevance Score:{score:.4f}")
if __name__ == '__main__':
    main()