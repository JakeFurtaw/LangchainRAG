import argparse
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

CHROMA_PATH = 'chroma'
MIXTRAL_CHAT_TEMPLATE = (
    "<s> [INST] <<SYS>>" 
    "You are a chatbot, designed to help new and existing students with any questions they may have about the university. "
    "You are programmed to be helpful, friendly, and informative. "
    "<</SYS>>"
    "[/INST]{answer_str} </s>"

    "<s>[INST] {question_str} <</INST>>"
)

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto")