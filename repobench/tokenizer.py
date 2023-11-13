import tiktoken
from transformers import AutoTokenizer


class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")
    
    def encode(self, text):
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

class CodeGenTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-6B-mono')

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
