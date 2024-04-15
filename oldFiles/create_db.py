"""
This script is used to create a database of text chunks from the .txt files in the data folder.
It uses the langchain library to load douments, split text, embed the text, and save the text to a Chroma database.
Then it creates a database of text chunks and saves them to a Chroma database.
"""
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
import torch

DATA_PATH = 'data'
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

#Set device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    documents = load_docs()
    chunks = split_pages(documents)
    save_to_db(chunks)
#Load the .txt files
def load_docs():
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    doc_text = [doc.page_content for doc in documents]
    return doc_text
#Split the documents into chunks of text
def split_pages(doc_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=10, 
        length_function=len,
    )
    chunks = text_splitter.create_documents(doc_text)
    return chunks
#Save files to database
def save_to_db(chunks):
    #clear db if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    #Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings.embedding_model.to(device)
    #Create new db from current docs
    db= Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
if __name__ == '__main__':
    main()