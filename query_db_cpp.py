from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain_community.vectorstores import Chroma
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

n_gpu_layers = 32  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


embed_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

llm = LlamaCpp(
    model_path="models/llama-2-70b-chat.Q5_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False,
)


db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_model)

if __name__ == "__main__":
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=db.as_retriever()
    )

    rag_pipeline("Who is Jal Irani?")
    print(rag_pipeline)