from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#path to the data
DATA_PATH = 'data'

#load the documents from .txt files
def load_docs():
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    return documents()

#split the documents into chunks of text
def split_pages(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separator='\n',
        chunk_size=100,
        chunk_overlap=25,
        length_function=len,

    )
    chunks = text_splitter.split(documents)
    return chunks


