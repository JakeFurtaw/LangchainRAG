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
import sys
from dotenv import load_dotenv
from pathlib import Path

# Path to the Chroma database
CHROMA_PATH = 'TowsonDB'
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# Chat template to get better results from the model
CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are an AI Assistant that helps college students navigate a college campus."
    "You provide information like teacher and faculty contact information, teachers office room numbers, course information, enrollment information, campus resources," 
    "and general campus information."
    "Please ensure that your responses are clear, concise, and positive in nature."
    "If you dont know the answer to a question, you can say that you are not sure."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
)
# Specify the GPU as device if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Load your Hugging Face API token
load_dotenv(Path(".env"))
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
#Configure the quantization config
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
# Load the Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("alpindale/WizardLM-2-8x22B", 
    quantization_config=quantization_config,
    chat_template=CHAT_TEMPLATE,
    device_map="auto")
model = AutoModelForCausalLM.from_pretrained("alpindale/WizardLM-2-8x22B", 
    quantization_config=quantization_config,
    device_map="auto")
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
        # Print the wrapped lines with a blank lines between each result
        print('\n'.join(wrapped_lines))
        print(f"Relevance Score: {score:.4f}")
        print('-' * 80)

# Main Function
def main():
    # Load the database and the embedding function
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, 
                embedding_function=embeddings)
    # Query the database
    if len(sys.argv) > 1:
        # Get the query
        query = ' '.join(sys.argv[1:])
        # Query the db for the most similar results
        results = db.similarity_search_with_relevance_scores(query, k=5)
        # Get the context for the chat prompt
        docs = []
        for result in results:
            document, score = result
            docs.append(document.page_content.strip())
        # Move the input tensors to the device
        input_tensors = tokenizer(query,
                                  return_tensors="pt",
                                  padding=True).to(device)
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
        print_results([(f"\n"+response_text, 1.0)])
    else:
        print("Please provide a query.")

if __name__ == '__main__':
    main()