"""Physics-informed quantum Hamiltonian layer (PennyLane + PyTorch).

A trainable parameterized Hamiltonian

    H(c) = sum_k c_k * P_k ,   P_k in {Z_i, Z_i Z_(i+1)}

is measured on a strongly-entangling variational circuit. The expectation
value <H> is exposed to the outside as a physics regularizer (we drive the
encoder toward low-energy, physically consistent states), while the local
Pauli expectations are projected back into the residual stream of the
transformer.
"""
import torch
import torch.nn as nn
import pennylane as qml


class QuantumHamiltonianLayer(nn.Module):
    def __init__(self, d_model, n_qubits=4, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.in_proj = nn.Linear(d_model, n_qubits)
        self.out_proj = nn.Linear(n_qubits + 1, d_model)
        self.out_proj.weight.data.mul_(0.1)  # start as a gentle perturbation
        self.weights = nn.Parameter(0.3 * torch.randn(n_layers, n_qubits, 3))
        local = [qml.PauliZ(i) for i in range(n_qubits)]
        coupling = [qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits) for i in range(n_qubits)]
        self._ops = local + coupling
        self.coeffs = nn.Parameter(0.2 * torch.randn(len(self._ops)))
        dev = qml.device("default.qubit", wires=n_qubits)
        self._circuit = self._build_qnode(dev)

    def _make_qnode(self, dev, diff_method):
        ops = self._ops
        n_qubits = self.n_qubits

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights, coeffs):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            H = qml.Hamiltonian(list(coeffs), ops)
            return [qml.expval(H)] + [qml.expval(op) for op in ops[:n_qubits]]

        return circuit

    def _build_qnode(self, dev):
        for diff in ("backprop", "parameter-shift", "finite-diff"):
            try:
                qn = self._make_qnode(dev, diff)
                out = qn(torch.zeros(self.n_qubits), self.weights, self.coeffs)
                out = torch.stack(list(out)) if isinstance(out, (list, tuple)) else out
                out.sum().backward()
                return qn
            except Exception:
                continue
        raise RuntimeError("no working PennyLane differentiation method found")

    def forward(self, x):
        """x: (B, d_model) pooled features -> (quantum features (B, d_model), energies (B,))"""
        angles = 2.0 * torch.arctan(self.in_proj(x))  # squash R -> (-pi, pi)
        feats = []
        for i in range(x.shape[0]):
            out = self._circuit(angles[i], self.weights, self.coeffs)
            feats.append(torch.stack(list(out)) if isinstance(out, (list, tuple)) else out)
        feats = torch.stack(feats).float()  # PennyLane emits float64; match trunk dtype
        return self.out_proj(feats), feats[:, 0]
