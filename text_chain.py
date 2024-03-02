from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_embeddings import OpenAIEmbeddings

#path to the data
DATA_PATH = 'data'
CHROMA_PATH = 'chroma'

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

    db= Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
