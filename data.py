"""Multimodal pre-training data: secure-code, cybersecurity defense and
long-horizon agentic workflows, paired with synthetic images and telemetry."""
import numpy as np
import torch

from .model import BOS, EOS, IMG, SEP, TEL, UNK

# --- tiny char-level tokenizer -------------------------------------------
ALPHABET = SPECIAL_CHARS = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " .,:;!?+-*/=(){}[]<>_#'\"\\\n\t%$&|@^~`"
)
_ITOS = list(ALPHABET)
_STOI = {ch: i for i, ch in enumerate(_ITOS)}


def _special_count():
    from .model import SPECIALS
    return len(SPECIALS)


VOCAB_SIZE = _special_count() + len(_ITOS)


def encode(text):
    from .model import SPECIALS
    off = len(SPECIALS)
    return [off + _STOI.get(ch, _STOI["?"]) for ch in text]


def decode(ids):
    from .model import SPECIALS
    off = len(SPECIALS)
    inv = {off + v: k for k, v in _STOI.items()}
    sp = {i: s for i, s in enumerate(SPECIALS)}
    return "".join(sp.get(i, inv.get(i, "?")) for i in ids)


# --- corpus ----------------------------------------------------------------
CORPUS = [
    # secure code generation
    "<task> sanitize sql input </task> <code> def safe_query(db, uid): "
    "return db.execute('SELECT * FROM users WHERE id = ?', (int(uid),)) </code>",
    "<task> hash password </task> <code> import hashlib, os "
    "salt = os.urandom(16) "
    "dk = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, 200_000) "
    "return salt + dk </code>",
    "<task> avoid eval </task> <code> cfg = json.loads(raw) "
    "if set(cfg) <= {'host','port'}: connect(cfg) </code>",
    "<task> constant time compare </task> <code> import hmac "
    "return hmac.compare_digest(token_a, token_b) </code>",
    "<task> bound loop </task> <code> for attempt in range(3): "
    "resp = retry_request(req) if resp.ok: break </code>",
    # cybersecurity defense
    "<alert> brute force ssh </alert> <plan> rate limit source "
    "rotate keys alert soc quarantine host reimage verify logs </plan>",
    "<alert> beacon c2 traffic </alert> <plan> capture pcaps block domain "
    "hunt persistence review processes patch system monitor egress </plan>",
    "<alert> ransomware onset </alert> <plan> isolate vlan freeze backups "
    "collect forensics identify patient zero restore from clean image </plan>",
    "<alert> credential dump </alert> <plan> revoke tokens reset passwords "
    "audit oauth grants deploy edr sensor review privileged access </plan>",
    # long-horizon agentic / engineering workflow
    "<project> migrate service to zero downtime </project> <plan> "
    "audit dependencies build shadow cluster mirror traffic compare outputs "
    "flip load balancer gradually drain old nodes validate slos clean up </plan>",
    "<project> fuzz parser for exploits </project> <plan> seed corpus "
    "generate mutations run harness triage crashes minimize cases file reports </plan>",
    "<project> harden build pipeline </project> <plan> pin dependencies "
    "enable sbom sign artifacts add provenance verify on deploy </plan>",
    # multimodal references
    "<image> traffic heat map </image> <analysis> elevated east ingress "
    "spike at 03:00 correlated with scan pattern recommend firewall rule </analysis>",
    "<telemetry> cpu 96 disk 99 net spike </telemetry> <analysis> "
    "resource exhaustion incident suspected restart service and investigate leak </analysis>",
]


def synth_image(rng):
    """16x16 'network traffic heat map' with an anomaly band."""
    img = rng.random((16, 16), dtype=np.float32) * 0.3
    band = rng.integers(4, 12)
    img[band, :] += rng.random(16, dtype=np.float32) * 0.6
    return torch.from_numpy(np.clip(img, 0, 1))


def synth_telemetry(rng):
    """16-dim security telemetry vector (cpu, disk, net, error rates...)."""
    v = rng.random(16, dtype=np.float32)
    if rng.random() < 0.5:  # inject an anomaly
        v[rng.integers(0, 16)] += 2.0
    return torch.from_numpy(v / v.max())


def build_dataset(seq_len=96, n_samples=600, seed=0):
    """Returns a list of (embedding_inputs, target_ids) usable for supervised
    next-token training on the fused multimodal prefix."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        text = CORPUS[rng.integers(len(CORPUS))]
        ids = [BOS] + encode(text) + [EOS]
        ids = ids[: seq_len - 1]
        x = torch.tensor(ids[:-1])
        y = torch.tensor(ids[1:])
        samples.append((x, y))
    return samples


def conditioning(prefix_text, image=None, telemetry=None, rng=None):
    """Build the conditioning sequence pieces for inference."""
    rng = rng or np.random.default_rng(0)
    toks = [BOS]
    if image is not None:
        toks += [IMG]
    if telemetry is not None:
        toks += [TEL]
    toks += encode(prefix_text)
    return toks, image, telemetry
