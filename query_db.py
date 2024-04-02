"""
This script is used to query the database of text chunks created in create_db.py.
It uses the langchain library to load the model and tokenizer.
Then the script queries the database and returns the result or results depending on the k #.
MAKE SURE TO REIGNORE THE .env FILE AFTER USE
"""
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import LlamaForCausalLM, LlamaTokenizer
from textwrap import wrap
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Specify the GPU as device if available
device = [torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")]

# Load your Hugging Face API token
load_dotenv(Path(".env"))
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Load the LLama2 model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", 
    load_in_8bit=True,#Trying to impliment quantization
    device_map="auto")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", 
    load_in_8bit=True,#Trying to impliment quantization
    device_map="auto")

# Move the model to the primary device
model.to(device)

# Path to the Chroma database
CHROMA_PATH = 'chroma'

# Chat template to get better results from LLama2 model
LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are an AI Assistant that helps college students navigate a college campus."
    "You provide information like teacher contact information, office room numbers, class schedules, and general campus information."
    "Please ensure that your responses are clear, concise, and positive in nature."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
)

# Printing Results to the CLI
def print_results(results):
    if not results:
        print("Sorry, I couldn't find any relevant information for your query.")
        return
    print("\nResults:")
    print('-' * 80)
    for result in results:
        if isinstance(result[0], str):
            content = result[0].strip()
        else:
            content = result[0].page_content.strip()
        score = result[1]
        # Split the content into lines
        lines = content.split('\n')
        # Remove empty lines and leading/trailing whitespace
        lines = [line.strip() for line in lines if line.strip()]
        # Wrap long lines
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(wrap(line, width=80))
        # Print the wrapped lines with a blank line between each result
        print('\n\n'.join(wrapped_lines))
        print(f"Relevance Score: {score:.4f}")
        print('-' * 80)

# Main Function
def main():
    # Load the database and the embedding function
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Query the database
    if len(sys.argv) > 1:
        # Get the query
        query = ' '.join(sys.argv[1:])
        # Query the db for the most similar results
        results = db.similarity_search_with_relevance_scores(query, k=5)
        # Get the chat prompt template
        docs = []
        for result in results:
            document, score = result
            docs.append(document.page_content.strip())
        prompt = LLAMA_CHAT_TEMPLATE.format(context_str=', \n\n'.join(docs), query_str='\n\n'+query)
        # Move the input tensors to the device
        input_tensors = tokenizer(prompt, return_tensors="pt").to(device)
        # Generate the response from the LLama2 model
        response = model.module.generate(**input_tensors)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        # Print the results
        print('-' * 80)
        print(f"Query: {query}")
        print_results([(f"\n"+response_text, 1.0)])
    else:
        print("Please provide a query.")

if __name__ == '__main__':
    main()