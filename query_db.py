"""
This script is used to query the database of text chunks created in create_db.py.
It uses the langchain library to load the model and tokenizer.
Then the script queries the database and returns the result or results depending on the k #.
MAKE SURE TO REIGNORE THE .env FILE AFTER USE
"""
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.prompts import ChatPromptTemplate
from textwrap import wrap
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Specify the GPU devices to use
gpu_indices = [0, 1]
devices = [torch.device(f"cuda:{i}") for i in gpu_indices if torch.cuda.is_available()]
torch.cuda.empty_cache()

# Set the device for the model
model_device = devices[1]  # Set the first GPU as the primary device
tokenizer_device = devices[0]  # Set the second GPU as the tokenizer device

# Check if multiple GPUs are available
if len(devices) > 1:
    print(f"Using {len(devices)} GPUs: {', '.join(str(device) for device in devices)}")
else:
    print(f"Using single GPU: {model_device}")

# Load your Hugging Face API token
load_dotenv(Path(".env"))
HF_API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")

# Load the LLama2 model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_API_KEY)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_API_KEY)

# Set up DataParallel if multiple GPUs are available
if len(devices) > 1:
    model = nn.DataParallel(model, device_ids=gpu_indices)

# Move the model to the primary device
model.to(model_device)

# Path to the Chroma database
CHROMA_PATH = 'chroma'

# Chat template to get better results from LLama2 model
LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are an AI Assistant that helps college students navigate a college campus."
    "You provide information like teacher contact information, room numbers, class schedules, and campus information."
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
        print('\n'.join(wrapped_lines))
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
        results = db.similarity_search_with_relevance_scores(query, k=3)
        # Get the chat prompt template
        docs = []
        for result in results:
            document, score = result
            docs.append(document.page_content.strip())
        prompt = LLAMA_CHAT_TEMPLATE.format(context_str=', \n\n'.join(docs), query_str=query)
        # Move the input tensors to the device
        input_tensors = tokenizer(prompt, return_tensors="pt").to(model_device)
        # Generate the response from the LLama2 model
        response = model.module.generate(**input_tensors)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        # Print the results
        print('-' * 80)
        print(f"\nQuery: {query}")
        print_results([(f"\n\n"+response_text, 1.0)])
    else:
        print("Please provide a query.")

if __name__ == '__main__':
    main()