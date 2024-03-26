from langchain.vectorstores import Chroma
import re

CHROMA_PATH = 'chroma'
CLEANED_DB_PATH = 'cleaned_chroma'

# Load the Chroma database
chroma_db = Chroma(persist_directory= CHROMA_PATH)

# Define the pattern to remove
pattern = r'[\x80-\xff]|�|[� -�]'

# Iterate through the documents and remove the pattern
updated_docs = []
for doc in chroma_db.as_retriever():
    cleaned_content = re.sub(pattern, '', doc.page_content, flags=re.UNICODE)
    updated_doc = doc.copy(page_content=cleaned_content)
    updated_docs.append(updated_doc)

# Create a new Chroma database with the cleaned documents
new_chroma_db = Chroma.from_documents(updated_docs, chroma_db.embedding_function, persist_directory='CLEANED_DB_PATH')
new_chroma_db.persist()