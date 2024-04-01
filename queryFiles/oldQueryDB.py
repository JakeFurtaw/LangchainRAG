""" This script is used to query the database of text chunks created in create_db.py. 
It uses the langchain library to load the model and tokenizer. 
Then the script queries the database and returns the result or results depending on the k #. 
MAKE SURE TO REIGNORE THE .env FILE AFTER USE """
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, LlamaForCausalLM
from langchain.prompts import ChatPromptTemplate
import torch
from textwrap import wrap
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

#Making sure the device is set to the GPU
device = torch.device("cuda")
torch.cuda.empty_cache()
#print(f"Running on {device}")  Testing what device is being used
# Load your Hugging Face API token
load_dotenv(Path(".env"))
HF_API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")
# Load the LLama2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_API_KEY)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HF_API_KEY)
model.to(device)
# Path to the Chroma database
CHROMA_PATH = 'chroma'
# Chat template to get better results from LLama2 model
LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are a helpful, respectful and honest AI assistant to help college students navigate a college campus."
    "Always answer as helpfully as possible, while being safe."
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
            # If the result is a string, assume it's the response_text
            response_text = result[0]
            print(response_text)
        else:
            # If the result is a Document object, handle it as before
            document = result[0]
            score = result[1]
            content = document.page_content.strip()
            lines = content.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            wrapped_lines = []
            for line in lines:
                wrapped_lines.extend(wrap(line, width=80))
            print('\n'.join(wrapped_lines))
            print(f"Relevance Score: {score:.4f}")
        print('-' * 80)
# Main Function
def main():
    # Load the database and the embedding function
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    if len(sys.argv) > 1:
        # Get the query
        query = ' '.join(sys.argv[1:])
        # Query the db for the most similar results
        results = db.similarity_search_with_relevance_scores(query, k=3)
        # Get the chat prompt template
        prompt_template = ChatPromptTemplate.from_template(LLAMA_CHAT_TEMPLATE)
        context_str = '\n'.join([f"{result[0]}\nRelevance Score: {result[1]:.4f}" for result in results])
        prompt = prompt_template.format(context_str=context_str, query_str=query)
        # Move the input tensors to the device
        input_tensors = tokenizer(prompt, return_tensors="pt").to(device)
        # Generate the response from the LLama2 model
        response = model.generate(**input_tensors)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        # Print the results
        print('-' * 80)
        print(f"\nQuery: {query}")
        print_results([(f"\n"+response_text, 1.0)])
    else:
        print("Please provide a query.")

if __name__ == '__main__':
    main()