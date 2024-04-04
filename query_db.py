""" This script is used to query the database of text chunks created in create_db.py. 
It uses the langchain library to load the model and tokenizer. 
Then the script queries the database and returns the result or results depending on the k #. 
MAKE SURE TO REIGNORE THE .env FILE AFTER USE """
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.prompts import ChatPromptTemplate
import torch
from textwrap import wrap
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

#Making sure the device is set to the GPU
device = torch.device("cuda")
# Load your Hugging Face API token
load_dotenv(Path(".env"))
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
# Load the LLama2 model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model.to(device)
# Path to the Chroma database
CHROMA_PATH = 'chroma'
# Chat template to get better results from LLama2 model
LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are an AI Assistant that helps college students navigate a college campus."
    "You provide information like teacher contact information, teachers office room numbers, course information, enrollment information, campus resources," 
    "and general campus information."
    "Please ensure that your responses are clear, concise, and positive in nature."
    "If you dont know the answer to a question, you can say that you are not sure."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
)
# Printing Results to the CLI
def print_results(results):
    if not results:
        print("Sorry, I couldn't find any relevant information for your query.")
        return

    for i, result in enumerate(results, start=1):
        content = result[0]
        score = result[1]
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        print(f"Result {i}:")
        print("Content:")
        for line in lines:
            print(line)
        print(f"Relevance Score: {score:.4f}\n")
# Main Function
def main():
    # Load the database and the embedding function
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    if len(sys.argv) > 1:
        # Get the query
        query = ' '.join(sys.argv[1:])
        # Query the db for the most similar results
        results = db.similarity_search_with_relevance_scores(query, k=5)
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
        print(f"\n\nQuery: {query}")
        print_results([(f"\n\n"+response_text, 1.0)])
    else:
        print("Please provide a query.")

if __name__ == '__main__':
    main()