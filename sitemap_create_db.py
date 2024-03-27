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
import re
import shutil

SITEMAP_URL = 'https://www.towson.edu/sitemap.xml'
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "intfloat/e5-large-v2"
def main():
    documents = load_docs()
    chunks = split_pages(documents)
    save_to_db(chunks)
# Load the documents from the sitemap.xml file
def load_docs():
    loader = SitemapLoader(SITEMAP_URL, continue_on_failure=True)
    documents = loader.load()
    doc_text = [doc.page_content for doc in documents]
    """
    I have no idea if it will work off rip. Don't know what the datatypes are, I'm guessing doc_text is a list of strings.
    I would suggest going through this on the debugger. Try creating a small sitemap too, maybe local? pull a few html pages, not sure.
    Waiting for the whole sitemap to be processed will make it take forever to do this correctly. Maybe look online for example sitemaps that
    are small
    """
    pattern = r'[ -~]+'
    doc_list = []
    for doc in doc_text:
        matches = re.findall(doc)
        new_doc = ''.join(matches)
        doc_list.append(new_doc)
    return doc_text

# Split the documents into chunks of text
def split_pages(doc_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, 
        chunk_overlap=15, 
        length_function=len,
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