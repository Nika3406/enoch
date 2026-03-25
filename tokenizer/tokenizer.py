import sentencepiece as spm
from pathlib import Path

class Tokenizer:
    def __init__(self, model_path="tokenizer/enoch.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens):
        return self.sp.decode(tokens)


def train_tokenizer(input_file, vocab_size=32000):
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix="tokenizer/enoch",
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0
    )
