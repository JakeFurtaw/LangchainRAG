from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import shutil

#path to the data
DATA_PATH = 'data'
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "intfloat/e5-large-v2"

def main():
    documents = load_docs()
    chunks = split_pages(documents)
    save_to_db(chunks)

#load the .txt files
def load_docs():
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    return documents()

#split the documents into chunks of text
def split_pages(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=25, length_function=len
    )
    chunks = text_splitter.split(documents)

    return chunks

#save files to database
def save_to_db(chunks):
    #clear db if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    #create new db from current docs
    db= Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()

if __name__ == '__main__':
    main()