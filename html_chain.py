from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import HTMLHeaderTextSplitter

DATA_PATH = 'data'

def load_docs():
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    return documents()

def split_pages(documents):
    html_splitter = HTMLHeaderTextSplitter()
    pages = html_splitter.split(documents)
    return pages