"""Agentic loop: the trained model drafts secure code / response plans from
multimodal input; deterministic guardrails (static checks, allowlists) filter
generations; the loop re-prompts with feedback for up to `max_iters` rounds."""
import numpy as np
import torch

from .data import conditioning, decode, encode, synth_image, synth_telemetry
from .model import BOS, EOS, MultimodalPIQAGPT


BANNED = ("eval(", "exec(", "os.system", "shell=True", "md5(", "http://", "password=")
REQUIRED_PLAN_WORDS = ("plan", "task")


def static_guardrails(text):
    """Return list of violations found by deterministic checks."""
    v = []
    low = text.lower()
    for b in BANNED:
        if b in low:
            v.append(f"banned pattern: {b}")
    if any(w in low for w in REQUIRED_PLAN_WORDS) is False and "def " not in low:
        v.append("output is neither a plan nor code")
    return v


class PIQAAgent:
    def __init__(self, model, max_iters=3, log=print):
        self.model = model
        self.max_iters = max_iters
        self.log = log

    def think(self, prompt, image=None, telemetry=None, max_new=250, temperature=0.8):
        self.model.eval()
        feedback = None
        for it in range(self.max_iters):
            p = prompt if feedback is None else f"{prompt} <feedback> {feedback} </feedback>"
            toks, img, tel = conditioning(p, image, telemetry)
            ids = torch.tensor(toks)
            emb_parts = []
            if img is not None:
                emb_parts.append(self.model.encode_image(img))
            if tel is not None:
                emb_parts.append(self.model.tel_proj(tel.unsqueeze(0)).unsqueeze(1))
            emb_parts.append(self.model.tok_emb(ids).unsqueeze(0))
            emb = torch.cat(emb_parts, dim=1)
            with torch.no_grad():
                out = self.model.generate(emb, max_new=max_new, temperature=temperature)
            text = decode(out).strip()
            violations = static_guardrails(text)
            if not violations:
                self.log(f"[agent] round {it + 1}: accepted")
                return text, violations, it + 1
            feedback = "; ".join(violations)
            self.log(f"[agent] round {it + 1}: rejected ({feedback})")
        return text, violations, self.max_iters

    def multimodal_think(self, prompt, seed=0, with_image=True, with_telemetry=True):
        rng = np.random.default_rng(seed)
        image = synth_image(rng) if with_image else None
        telemetry = synth_telemetry(rng) if with_telemetry else None
        return self.think(prompt, image=image, telemetry=telemetry)


def load_or_train(path="piqa_gpt.pt", log=print):
    from .data import VOCAB_SIZE
    model = MultimodalPIQAGPT(VOCAB_SIZE, d_model=64, n_head=4, n_blocks=2,
                              ctx=448, use_quantum=True, n_qubits=4)
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        log(f"loaded weights from {path}")
    except FileNotFoundError:
        log("no checkpoint found, training from scratch...")
        from .train import train
        model = train(log=log)
        torch.save(model.state_dict(), path)
    return model


if __name__ == "__main__":
    model = load_or_train()
    agent = PIQAAgent(model)
    rng = np.random.default_rng(7)
    text, violations, rounds = agent.multimodal_think("<task> sanitize sql input </task>")
    print("OUTPUT:", text[:400])
    print("violations:", violations, "rounds:", rounds)
