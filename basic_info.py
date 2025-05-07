import torch
import gpt-2
import config


torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_124M)
total_params=sum(p.numel() for p in model.parameters())
print(total_params)