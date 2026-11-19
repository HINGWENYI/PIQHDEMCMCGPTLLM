"""GPT-style transformer trunk with a quantum Hamiltonian layer and multimodal
(image / telemetry / text) conditioning encoders."""
import math

import torch
import torch.nn as nn

from .quantum import QuantumHamiltonianLayer

SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<IMG>", "<TEL>"]
PAD, UNK, BOS, EOS, SEP, IMG, TEL = range(len(SPECIALS))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PIQAGPT(nn.Module):
    """Causal transformer whose residual stream is modulated by the measured
    expectation values of a parameterized quantum Hamiltonian."""

    def __init__(self, vocab_size, d_model=64, n_head=4, n_blocks=2, ctx=448,
                 use_quantum=True, n_qubits=4):
        super().__init__()
        self.d_model = d_model
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)  # tied head needs small init
        self.pos_emb = nn.Parameter(0.02 * torch.randn(ctx, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.qnn = QuantumHamiltonianLayer(d_model, n_qubits) if use_quantum else None

    def physics_energy(self, ids):
        """Bounded physics regularizer: <H> / sum|c_k| lies in [-1, 1] for any
        coefficient scale (each Pauli term has spectral norm 1), so the loss
        cannot be driven down without bound by inflating the Hamiltonian."""
        x = self.tok_emb(ids)
        energy = self.qnn(x.mean(dim=1).float())[1]
        norm = self.qnn.coeffs.abs().sum().detach() + 1e-3
        return energy / norm

    def trunk(self, embeds):
        """embeds: (B, T, d) raw (position-free) embeddings -> (logits, energies)."""
        T = embeds.size(1)
        x = embeds + self.pos_emb[:T].unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        energy = None
        if self.qnn is not None:
            q, energy = self.qnn(x.mean(dim=1))
            x = x + q.unsqueeze(1)
        return self.head(x), energy


class MultimodalPIQAGPT(PIQAGPT):
    """PIQAGPT + encoders that fuse image patches, security telemetry and text
    into one conditioning sequence."""

    def __init__(self, vocab_size, d_model=64, n_head=4, n_blocks=2, ctx=448,
                 use_quantum=True, n_qubits=4, img_size=16, patch=4, tel_dim=16):
        super().__init__(vocab_size, d_model, n_head, n_blocks, ctx, use_quantum, n_qubits)
        self.img_size = img_size
        self.patch = patch
        self.img_proj = nn.Linear(patch * patch, d_model)
        self.tel_proj = nn.Linear(tel_dim, d_model)

    def encode_image(self, img):
        """img: (1, 16, 16) -> (1, n_patches, d_model)."""
        p = self.patch
        n = self.img_size // p
        patches = img.unfold(1, p, p).unfold(2, p, p)
        patches = patches.reshape(1, n * n, p * p)
        return self.img_proj(patches)

    def encode_condition(self, prompt_ids, images=None, telemetry=None):
        """Build the (1, C, d_model) conditioning embedding sequence."""
        parts = []
        if images is not None:
            parts.append(self.encode_image(images))
        if telemetry is not None:
            parts.append(self.tel_proj(telemetry).unsqueeze(1))
        parts.append(self.tok_emb(prompt_ids).unsqueeze(0))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def generate(self, cond_embeds, max_new=300, temperature=0.8, top_k=4, eos_id=EOS):
        emb = cond_embeds
        out = []
        for _ in range(max_new):
            window = emb[:, -self.ctx:]
            logits, _ = self.trunk(window)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(logits.softmax(dim=-1), 1)
            if nxt.item() == eos_id:
                break
            out.append(nxt.item())
            emb = torch.cat([emb, self.tok_emb(nxt)], dim=1)
        return out
