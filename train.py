from tokenizer.tokenizer import Tokenizer, train_tokenizer
from data.dataset import TextDataset
from model.gpt import EnochGPT
from trainer.trainer import train_model

# 1) Train tokenizer once
# train_tokenizer("data/corpus.txt")

# 2) Load tokenizer
tok = Tokenizer()

# 3) Load corpus
text = open("data/corpus.txt","r",encoding="utf-8").read()
tokens = tok.encode(text)

# 4) Dataset
dataset = TextDataset(tokens)

# 5) Model
model = EnochGPT()

# 6) Train
train_model(model, dataset, epochs=3)
