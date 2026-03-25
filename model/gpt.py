import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    vocab_size = 32000
    block_size = 256
    n_layers = 8
    n_heads = 8
    n_embd = 512
    dropout = 0.1

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(cfg.n_embd, 4*cfg.n_embd),
            nn.GELU(),
            nn.Linear(4*cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout)
        )
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        attn_out,_ = self.attn(x,x,x,need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x

class EnochGPT(nn.Module):
    def __init__(self, cfg=GPTConfig()):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)

        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))

        return logits, loss
