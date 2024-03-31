"""
This script is used to create a database of text chunks from a stiemap.
It uses the langchain library to load douments, split text, embed the text, and save the text to a Chroma database.
Then it creates a database of text chunks and saves them to a Chroma database.
"""
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import shutil
from torch import cuda
import re

SITEMAP_URL_PATH = "https://www.towson.edu/sitemap.xml"
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


def run():
    documents = load_docs()
    chunks = split_pages(documents)
    save_to_db(chunks)


def load_docs():
    # Load the documents from the sitemap.xml file
    loader = SitemapLoader(SITEMAP_URL_PATH, continue_on_failure=True)
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


def split_pages(doc_text):
    # Split the documents into chunks of text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.create_documents(doc_text)
    return chunks


def save_to_db(chunks):
    # Save files to the database
    # Clear the database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # Create a new database from the current documents
    db = Chroma.from_documents(
        chunks, embed_model, persist_directory=CHROMA_PATH
    )
    db.persist()


if __name__ == '__main__':
    run()

