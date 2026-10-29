"""Does the memory state change an SQS neuron's spike probabilities?

For one input-output and one memory qubit (N = N_mem = 1), the script builds many random
neurons, puts each neuron's memory qubit in many random states, applies one time step to the
same input, and prints the largest spread (max - min over memory states) of any spike
probability. Three circuits, each under both angle initialisations (``uniform``: U(0, pi);
``small``: N(0, 0.1)):

* ``crx_all``, the published circuit: every two-qubit gate writes to the memory and none reads
  from it, so the spread is zero up to floating-point error (about 1e-15 in complex128);
* ``crx_both``, with the read gate: the memory controls a rotation of the input-output qubit,
  so the spread is far from zero;
* ``crx_both`` with the read-gate angles fixed at 0: the read gate is the identity, the circuit
  is ``crx_all`` again, and the spread is back to floating-point error.

The exit status is 1 if a circuit does not behave as listed.

    uv run python scripts/check_memory_blindness.py
"""
from __future__ import annotations

import sys

import torch

from sqs_bench.neurons.sqs import SQSLayer

N_NEURONS = 512   # random neurons (independent angle and weight draws)
N_STATES = 64     # random memory states per neuron
BLIND = 1e-12     # a spread below this is floating-point error
CIRCUITS = [  # (label, ansatz, extra layer arguments, blind to the memory?)
    ("crx_all (published circuit)", "crx_all", {}, True),
    ("crx_both (read gate)", "crx_both", {}, False),
    ("crx_both, read-gate angle 0", "crx_both", {"theta_crx2_scale": 0.0, "theta_crx2_trainable": False}, True),
]


def max_spread(ansatz: str, theta_init: str, extra: dict, seed: int = 0) -> float:
    torch.manual_seed(seed)
    layer = SQSLayer(3, N_NEURONS, 1, 1, ansatz=ansatz, theta_init=theta_init, dtype=torch.complex128, **extra)
    with torch.no_grad():
        layer.weight.abs_().add_(0.3)  # positive input currents, so the input rotation is not the identity
        x = torch.rand(1, 1, 3, dtype=torch.float64).expand(N_STATES, 1, 3)  # one input for every state
        memory = torch.randn(N_STATES, N_NEURONS, 2, dtype=torch.complex128)
        memory = memory / memory.norm(dim=-1, keepdim=True)
        # big-endian, input-output qubit first: |0>_io (x) |m> fills the first two amplitudes
        psi = torch.cat([memory, torch.zeros_like(memory)], dim=-1)
        p_joint, _ = layer.probabilities(x, psi)  # [state, neuron, outcome]
    return (p_joint.amax(dim=0) - p_joint.amin(dim=0)).max().item()


def main() -> int:
    print(f"N = N_mem = 1, complex128; {N_NEURONS} random neurons x {N_STATES} random memory states each\n")
    print(f"      {'circuit':<30} {'angle init':<11} largest spread of a spike probability")
    failed = 0
    for label, ansatz, extra, blind in CIRCUITS:
        for theta_init in ("uniform", "small"):
            spread = max_spread(ansatz, theta_init, extra)
            ok = (spread < BLIND) == blind
            failed += not ok
            expected = "floating-point error" if blind else "depends on the memory"
            print(f"{'ok' if ok else 'FAIL':>4}  {label:<30} {theta_init:<11} {spread:.2e}  (expected: {expected})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
