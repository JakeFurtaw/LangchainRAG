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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
load_dotenv(Path(".env"))
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.float16)

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
    response_text = results[0]
    print("\n"+response_text)
    print('-' * 80)
# Main Function
def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    while True:
        query = input("Enter query: ")
        if len(query) == 0:
            print("Please enter a query.")
            continue
        elif query.lower() == "exit":
            break

        search_results = db.similarity_search_with_relevance_scores(query, k=3)
        docs = []
        documents = []
        for result in search_results:
            document, score = result
            docs.append(document.page_content.strip())
            print(f"Database Results:\n {document.page_content.strip()}")
            print(f"Relevance score: {score}")
            print("-" * 80)
            documents.append(document.page_content.strip())

        context_str = query + "\n\n".join(documents)
        input_tensors = tokenizer(context_str, return_tensors="pt", padding=True).to(device)
        response = model.generate(**input_tensors, max_new_tokens=512, temperature=0.3, do_sample=True)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)

        print('-' * 80)
        print(f"Query: {query}")
        print_results([(f"\n"+response_text)])

if __name__ == '__main__':
    main()