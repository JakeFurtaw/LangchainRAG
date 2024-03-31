import re
from transformers import AutoModel

if __name__ == "__main__":
    # Load model directly
    model = AutoModel.from_pretrained("TheBloke/Llama-2-13B-chat-GGUF")