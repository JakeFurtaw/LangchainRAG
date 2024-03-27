from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import re

CHROMA_PATH = 'chroma'
CLEANED_DB_PATH = 'cleaned_chroma'

# Load the Chroma database
chroma_db = Chroma(persist_directory=CHROMA_PATH)

# Define the pattern
pattern = r'[^a-zA-Z0-9\s\n]'

# Iterate through the documents and remove everything except the pattern
updated_docs = []
for doc_tuple in chroma_db.as_retriever():
    doc_content, doc_score = doc_tuple
    cleaned_content = re.sub(pattern, '', doc_content, flags=re.UNICODE)
    metadata = {'score': str(doc_score)}
    # Convert score to string
    updated_doc = Document(page_content=cleaned_content, metadata=metadata)
    updated_docs.append(updated_doc)

# Create a new Chroma database with the cleaned documents
new_chroma_db = Chroma.from_documents(
    updated_docs, HuggingFaceEmbeddings(), persist_directory=CLEANED_DB_PATH)
new_chroma_db.persist()