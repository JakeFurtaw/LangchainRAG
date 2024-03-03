"""
This script is used to query the database of text chunks created in create_db.py.
It uses the langchain library to load the database, query the database, and return the most similar text chunks.
"""
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

CHROMA_PATH = 'chroma'
#chat template to get better results from mixtral
MIXTRAL_CHAT_TEMPLATE = (
    "<s> [INST] <<SYS>>" 
    "You are a chatbot, designed to help new and existing students with any questions they may have about the university. "
    "You are programmed to be helpful, friendly, and informative. "
    "<</SYS>>"
    "[/INST]{answer_str} </s>"

    "<s>[INST] {question_str} <</INST>>"
)

#load the mixtral model
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto")