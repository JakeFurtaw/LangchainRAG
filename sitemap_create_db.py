"""
This script is used to create a database of text chunks from a stiemap.
It uses the langchain library to load douments, split text, embed the text, and save the text to a Chroma database.
Then it creates a database of text chunks and saves them to a Chroma database.
"""
from langchain_community.document_loaders import SitemapLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
import requests

SITEMAP_URL = 'https://www.towson.edu/sitemap.xml'
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "intfloat/e5-large-v2"

def main():
    documents = load_docs()
    chunks = split_pages(documents)
    save_to_db(chunks)

# Load the documents from the sitemap.xml file to create vector database
def load_docs():
    loader = SitemapLoader(SITEMAP_URL, encoding='latin-1', continue_on_failure=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from the sitemap")

    # Load PDF files
    pdf_docs = []
    for doc in documents:
        if doc.metadata.source.endswith('.pdf'):
            try:
                response = requests.get(doc.metadata.source)
                pdf_loader = UnstructuredFileLoader(files=response.content, file_type='pdf')
                pdf_docs.extend(pdf_loader.load())
            except Exception as e:
                print(f"Error loading PDF file: {doc.metadata.source} - {e}")

    doc_text = [doc.page_content for doc in documents] + [doc.page_content for doc in pdf_docs]
    print(f"Loaded a total of {len(doc_text)} documents")
    return doc_text

# Split the documents into chunks of text
def split_pages(doc_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=15, length_function=len,
    )
    chunks = text_splitter.create_documents(doc_text)
    return chunks

# Save files to the database
def save_to_db(chunks):
    # Clear the database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # Create a new database from the current documents
    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()

if __name__ == '__main__':
    main()