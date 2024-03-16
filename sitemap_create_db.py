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
import aiohttp
import asyncio

SITEMAP_URL = 'https://www.towson.edu/sitemap.xml'
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "intfloat/e5-large-v2"
def main():
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    documents = loop.run_until_complete(load_docs())
    chunks = split_pages(documents)
    save_to_db(chunks)
# Load the documents from the sitemap.xml file
async def load_docs():
    loader = SitemapLoader(SITEMAP_URL)
    urls = [url for url in loader.sitemap_urls if not url.endswith(".pdf")]
    doc_texts = []
    for url in urls:
        try:
            doc = await loader._fetch_with_rate_limit(url)
            doc_text = doc.page_content
            doc_texts.append(doc_text)
        except (UnicodeDecodeError, aiohttp.ContentTypeError):
            # Skip non-text files
            continue
    return doc_texts
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