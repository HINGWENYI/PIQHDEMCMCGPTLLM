"""Post-variational refinement: Differential Evolution proposes parameter
updates, Metropolis MCMC accepts/rejects them against an energy that mixes
NLL, the quantum Hamiltonian expectation and a secure-code risk penalty.

Used after (or alongside) gradient descent, mirroring post-variational
estimates where classical optimization refines around the variational optimum.
"""
import numpy as np
import torch


def _as_vector(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def _write_vector(params, vec):
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p).data)
        i += n


def evaluate_energy(model, batch, lam_physics=0.05, risk_fn=None, lam_risk=0.5):
    """Lower is better: NLL + lam_physics*<H> + lam_risk*risk. batch = (x, y)."""
    model.eval()
    loss_dict = {}
    x, y = batch
    with torch.no_grad():
        logits, _ = model.trunk(model.tok_emb(x))
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
        )
        if hasattr(model, "physics_energy") and model.qnn is not None and lam_physics:
            physics = model.physics_energy(x).mean()
        else:
            physics = torch.tensor(0.0)
        if risk_fn is not None:
            risk = torch.tensor(float(risk_fn(logits)))
        else:
            risk = torch.tensor(0.0)
    loss_dict = {"nll": nll.item(), "physics": float(physics), "risk": float(risk)}
    E = loss_dict["nll"] + lam_physics * loss_dict["physics"] + lam_risk * loss_dict["risk"]
    return E, loss_dict


def demc_refine(model, batches, lam_physics=0.05, risk_fn=None, lam_risk=0.5,
                pop_size=8, generations=4, beta=2.0, seed=0, log=print):
    """DE proposals + Metropolis acceptance. `batches` yield (x, y) pairs whose
    forward signature matches model(x) -> (logits, energy)."""
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    theta0 = _as_vector(params).cpu().numpy()
    dim = theta0.size

    def energy_at(theta):
        _write_vector(params, torch.from_numpy(theta).float())
        E, parts = evaluate_energy(model, next(iter(batches())), lam_physics, risk_fn, lam_risk)
        return E, parts

    E0, parts0 = energy_at(theta0)
    log(f"[DE-MCMC] init energy {E0:.4f} {parts0}")
    pop = [theta0] + [theta0 + 0.05 * rng.standard_normal(dim) for _ in range(pop_size - 1)]
    pop_E = []
    for ind in pop[1:]:
        _write_vector(params, torch.from_numpy(ind).float())
        E, _ = evaluate_energy(model, next(iter(batches())), lam_physics, risk_fn, lam_risk)
        pop_E.append(E)
    pop_E = [E0] + pop_E
    best = int(np.argmin(pop_E))

    for g in range(generations):
        for i in range(pop_size):
            a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            F = 0.5 + 0.3 * rng.random()
            trial = pop[a] + F * (pop[b] - pop[c])
            cross = rng.random(dim) < 0.9
            cross[rng.integers(dim)] = True
            trial = np.where(cross, trial, pop[i])
            _write_vector(params, torch.from_numpy(trial).float())
            E_t, _ = evaluate_energy(model, next(iter(batches())), lam_physics, risk_fn, lam_risk)
            # Metropolis: always accept improvements, otherwise with prob exp(-beta*dE)
            dE = E_t - pop_E[i]
            if dE < 0 or rng.random() < np.exp(-beta * max(dE, 1e-12)):
                pop[i], pop_E[i] = trial, E_t
                if E_t < pop_E[best]:
                    best = i
        log(f"[DE-MCMC] gen {g + 1}/{generations} best {min(pop_E):.4f} mean {float(np.mean(pop_E)):.4f}")

    _write_vector(params, torch.from_numpy(pop[best]).float())
    return pop[best], min(pop_E)
