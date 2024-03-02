from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    doc_text = [doc.page_content for doc in documents]
    return doc_text

#split the documents into chunks of text
def split_pages(doc_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=25, length_function=len,
    )
    chunks = text_splitter.create_documents(doc_text)
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
    print(chunks[47:50])
    db.persist()

if __name__ == '__main__':
    main()