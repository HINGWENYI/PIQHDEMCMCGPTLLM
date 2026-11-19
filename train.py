"""Pre-training loop: AdamW with a physics-informed Hamiltonian regularizer
and a secure-code risk penalty, followed by post-variational DE-MCMC polish.
"""
import math
import time

import torch

from .data import build_dataset
from .demc import demc_refine
from .model import EOS, MultimodalPIQAGPT


RISKY_SUBSTRINGS = ("eval(", "exec(", "os.system", "shell=True", "password=")


def secure_code_risk(logits):
    """Penalize logits that favor tokens of risky substrings (char-level proxy)."""
    from .data import ALPHABET, _STOI
    off = 7  # number of special tokens
    ids = [off + _STOI[c] for s in RISKY_SUBSTRINGS for c in s if c in _STOI]
    return logits[..., ids].logsumexp(dim=-1).mean() / len(ids)


def make_model(vocab_size):
    return MultimodalPIQAGPT(vocab_size, d_model=64, n_head=4, n_blocks=2,
                             ctx=448, use_quantum=True, n_qubits=4)


def batched(samples, bs, ctx=440):
    """Stream-style batches: samples are concatenated and cut into ctx-sized
    windows, so the model never sees padding tokens."""
    stream_x = torch.cat([x for x, _ in samples])
    stream_y = torch.cat([y for _, y in samples])
    for i in range(0, stream_x.size(0) - 1, ctx):
        x = stream_x[i:i + ctx].unsqueeze(0)
        y = stream_y[i:i + ctx].unsqueeze(0)
        if x.size(1) >= 2:
            yield x, y


def forward_batch(model, batch):
    x, y = batch
    logits, energy = model.trunk(model.tok_emb(x))
    return logits, energy, y


def train(epochs=2, n_samples=600, bs=32, lr=3e-4, lam_physics=0.05,
          lam_risk=0.0, refine=True, log=print):
    """lam_risk stays 0 by default: the char-level proxy penalizes common
    letters inside risky substrings, which fights language modeling. Secure-code
    preferences come from the curated corpus + agent guardrails instead."""
    torch.manual_seed(0)
    from .data import VOCAB_SIZE
    model = make_model(VOCAB_SIZE)
    data = build_dataset(n_samples=n_samples)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for batch in batched(data, bs):
            x, y = batch
            logits, _ = model.trunk(model.tok_emb(x))
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
            )
            if model.qnn is not None and lam_physics:
                loss = loss + lam_physics * model.physics_energy(x).mean()
            if lam_risk:
                loss = loss + lam_risk * secure_code_risk(logits)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
        log(f"epoch {ep + 1}/{epochs} loss {tot / n:.4f} ({time.time() - t0:.1f}s)")
    if refine:
        demc_refine(
            model,
            lambda: batched(data[:64], 16),
            lam_physics=lam_physics,
            risk_fn=secure_code_risk,
            lam_risk=lam_risk,
            pop_size=6,
            generations=3,
            log=log,
        )
    return model


if __name__ == "__main__":
    model = train()
    torch.save(model.state_dict(), "piqa_gpt.pt")
    print("saved piqa_gpt.pt")
