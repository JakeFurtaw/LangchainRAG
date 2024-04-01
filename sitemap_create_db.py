"""
This script is used to create a database of text chunks from a stiemap.
It uses the langchain library to load douments, split text, embed the text, and save the text to a Chroma database.
Then it creates a database of text chunks and saves them to a Chroma database.
"""
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
import re
import torch

SITEMAP_URL = 'https://www.towson.edu/sitemap.xml'
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "intfloat/e5-large-v2"

# Set device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    documents = load_docs()
    chunks = split_pages(documents)
    save_to_db(chunks)

# Load the documents from the sitemap.xml file
def load_docs():
    loader = SitemapLoader(SITEMAP_URL, continue_on_failure=True)
    documents = loader.load()
    doc_text = [doc.page_content for doc in documents]
    docs = []
    for doc in doc_text:
        # Remove extra white space
        cleaned_text = re.sub(r'[ \t]+', ' ', doc)
        # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
        docs.append(cleaned_text)
    return docs
# Split the documents into chunks of text
def split_pages(doc_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.create_documents(doc_text)
    return chunks
# Save files to the database
def save_to_db(chunks):
    # Clear the database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings.embedding_model.to(device)

    # Create a new database from the current documents
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()

if __name__ == '__main__':
    main()