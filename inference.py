import torch
from device import DEVICE
from tokenizer.tokenizer import Tokenizer
from model.gpt import EnochGPT

@torch.no_grad()
def generate(model, idx, max_new_tokens=100):
    model.eval()
    for _ in range(max_new_tokens):
        logits,_ = model(idx)
        probs = torch.softmax(logits[:,-1,:], dim=-1)
        next_token = torch.multinomial(probs,1)
        idx = torch.cat([idx,next_token],dim=1)
    return idx

tok = Tokenizer()
model = EnochGPT()
model.load_state_dict(torch.load("enoch.pt", map_location=DEVICE))
model.to(DEVICE)

prompt = "In the beginning"
tokens = tok.encode(prompt)
idx = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

out = generate(model, idx)
print(tok.decode(out[0].tolist()))
