import torch
import tiktoken
from gpt2 import GPTModel, generate_text_simple
from config import GPT_CONFIG_124M_2
from tokenization import create_datalaoder_v1

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M_2)

def text_to_token_ids(text, tokenizer):
    encoded=tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor=torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat=token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context="Every effort moves you"
tokenizer=tiktoken.get_encoding("gpt2")

token_ids=generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M_2["context_length"]
)

print(token_ids_to_text(token_ids, tokenizer))