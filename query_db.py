"""
This script is used to query the database of text chunks created in create_db.py.
It uses the langchain library to load the model and tokenizer.
Then the script queries the database and returns the result or results depending on the k #.
"""
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from textwrap import wrap
import os
import sys
from dotenv import load_dotenv
# Load the Hugging Face API token
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
login(token=HUGGINGFACE_API_TOKEN)
# Load the LLaMA model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HUGGINGFACE_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=HUGGINGFACE_API_TOKEN)
CHROMA_PATH = 'chroma'
# Chat template to get better results from LLama model
LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are a helpful, respectful and honest AI assistant to help college students navigate a college campus."
    "Always answer as helpfully as possible, while being safe."
    "Please ensure that your responses are clear, concise, and positive in nature."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
)
def print_results(results):
    if not results:
        print("No results found.")
        return
    print("\nResults:")
    print('-' * 80)
    for result in results:
        document, score = result
        content = document.page_content.strip()
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

def main():
    # Load the database
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # Query the database
    if len(sys.argv) > 1:
        # Get the query
        query = ' '.join(sys.argv[1:])
        # Query the db for the most similar results
        results = db.similarity_search_with_relevance_scores(query, k=2)
        # Print the results
        print('-' * 80)
        print(f"\nQuery: {query}")
        print_results(results)
    else:
        print("Please provide a query.")

if __name__ == '__main__':
    main()