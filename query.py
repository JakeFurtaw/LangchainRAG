from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

CHROMA_PATH = 'TowsonDB'
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
CONVERSATION_HISTORY = []
CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    "You are an AI Assistant that helps college students navigate Towson University campus. "
    "Provide factual information based solely on the context given from the university's website. "
    "Give students information on campus facilities, services, and academic programs, teachers and their office locations, emails. "
    "Do not speculate or make up information if it is not covered in the context. "
    "Respond with clear, concise, and focused answers directly addressing the query. "
    "Use a positive and respectful tone suitable for college students. "
    "If you do not have enough information to answer a query, politely state that you are unable to provide a satisfactory answer."
    "{conversation_history} {context_str} <|eot_id|><|start_header_id|>user<|end_header_id|> {query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv(Path(".env"))
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization_config, device_map="auto")

def print_results(query, response_text):
    if not response_text:
        print("Sorry, I couldn't find any relevant information for your query.")
        return
    CONVERSATION_HISTORY.append(f"Query: {query}")
    CONVERSATION_HISTORY.append(f"Response: {response_text}")
    print(f"Query: {query}")
    print("\nResponse:")
    print('-' * 80)
    print(response_text)
    print('-' * 80)

def get_relevant_documents(query, db):
    search_results = db.similarity_search_with_relevance_scores(query, k=5)
    docs = []
    for result in search_results:
        document, score = result
        docs.append(document.page_content.strip())
        print(f"Database Results:\n {document.page_content.strip()}")
        print(f"Relevance score: {score}")
        print("-" * 80)
    return docs

def generate_response(query, context_str):
    conversation_history = "\n".join(CONVERSATION_HISTORY)
    input_text = CHAT_TEMPLATE.format(conversation_history=conversation_history, context_str=context_str, query=query)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_tensors = tokenizer.encode(input_text, return_tensors="pt", padding=True).to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    output = model.generate(input_ids=input_tensors, max_new_tokens=256, repetition_penalty=1.2, top_p=.95, temperature=0.1, do_sample=True)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return response_text

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

        docs = get_relevant_documents(query, db)
        context_str = "\n\n".join(docs)
        response_text = generate_response(query, context_str)
        print_results(query, response_text)

if __name__ == '__main__':
    main()