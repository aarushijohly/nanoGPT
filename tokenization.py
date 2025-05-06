import os
import re


#implementing a simple text tokenizer

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {j:i for i, j in vocab.items()}

    def encode(self, text):
        preprecessed = re.split(r'[,.?_!"()\']|--|\\s', text)
        preprecessed = [item.strip() for item in     ]