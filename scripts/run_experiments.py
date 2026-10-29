"""Train every run behind results/runs.csv, one experiment set (a "tier" here) at a time.

Each tier is one Hydra multirun of ``sqs-bench`` over models x delays x seeds, launched with
Hydra's basic launcher, so the runs go one after another. Run directories are
``<out_root>/sqs_ce/memprobe/<model>_<tier>_delay<d>/seed<k>/``.

    uv run python scripts/run_experiments.py --tier sweep60 --dry-run
    uv run python scripts/run_experiments.py --tier plateau --seeds 0-1
    uv run python scripts/run_experiments.py --tier all train.epochs=1 out_root=outputs/quick

Arguments after the options are Hydra overrides. Each replaces the tier's own value for the
same key (``train.epochs=1`` for a quick check, ``out_root=...`` to write somewhere else) or
is added to the command. Finished runs are skipped (``skip_if_done=true``), so a tier that
stopped part way can be started again. A run counts as finished when its ``metrics.json``
exists, whatever its settings, so give quick checks their own ``out_root``.

On the CPU the runs use two threads (``OMP_NUM_THREADS=2``, ``MKL_NUM_THREADS=2``), as the
recorded sweep60 runs did; set either variable yourself to change it. A tier that defaults
to CUDA falls back to the CPU, with a note, when torch finds no CUDA device and ``--device``
is not given.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

COMMON = ["protocol=sqs_ce", "dataset=memprobe", "skip_if_done=true"]

TIERS = {
    # 60 epochs, the cue on the first step, on the CPU
    "sweep60": {
        "models": ["qlif", "sqsnn"],
        "delays": [1, 2, 4, 8, 16],
        "seeds": [0, 1, 2],
        "device": "cpu",
        "overrides": ["train.epochs=60"],
    },
    # 150 epochs, one noise step before the cue
    "long150": {
        "models": ["sqsnn", "sqsnn_read"],
        "delays": [1, 2, 4, 8],
        "seeds": [0, 1, 2, 3, 4],
        "device": "cuda",
        "overrides": ["dataset.lead_in=1", "train.epochs=150"],
    },
    # at most 300 epochs: stop once the train loss has not dropped by more than 1e-4 for
    # 60 epochs; test sequences that also occur in the training set are replaced
    "plateau": {
        "models": ["sqsnn", "sqsnn_read"],
        "delays": [8, 16],
        "seeds": [0, 1, 2, 3, 4],
        "device": "cuda",
        "overrides": [
            "dataset.lead_in=1",
            "dataset.dedupe_test=true",
            "train=surrogate",  # the default trainer; the recorded runs named it explicitly
            "train.epochs=300",
            "train.patience=60",
            "train.min_delta=0.0001",
        ],
    },
    # read gate on or off, crossed with small or uniform initial angles
    "factorial": {
        "models": ["sqsnn", "sqsnn_read", "sqsnn_uni", "sqsnn_read_uni"],
        "delays": [1, 2],
        "seeds": list(range(10)),
        "device": "cuda",
        "overrides": ["dataset.lead_in=1", "train.epochs=60"],
    },
    # read-gate angles frozen at their initial draw from U(0, a), a = 0, pi/4, pi/2, pi
    "angle": {
        "models": ["sqsnn_angle0", "sqsnn_angle_pi4", "sqsnn_angle_pi2", "sqsnn_angle_pi"],
        "delays": [1, 2],
        "seeds": list(range(8)),
        "device": "cuda",
        "overrides": [
            "dataset.lead_in=1",
            "dataset.dedupe_test=true",
            "train=surrogate",  # the default trainer; the recorded runs named it explicitly
            "train.epochs=60",
        ],
    },
}


def parse_seeds(text: str) -> list[int]:
    """``"0-4"`` -> [0, 1, 2, 3, 4]; ``"0,3,7"`` -> [0, 3, 7]."""
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def cuda_available() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def join(values) -> str:
    return ",".join(str(v) for v in values)


def tier_overrides(name: str, device: str, seeds: list[int] | None, extra: list[str]) -> list[str]:
    """The Hydra overrides of one tier; ``extra`` replaces entries with the same key."""
    tier = TIERS[name]
    overrides = COMMON + [
        f"device={device}",
        f"model={join(tier['models'])}",
        f"dataset.delay={join(tier['delays'])}",
        # resolved in each run, so the tag always names that run's own delay
        f"variant={name}_delay${{dataset.delay}}",
        f"seed={join(seeds or tier['seeds'])}",
    ] + tier["overrides"]
    merged = {item.partition("=")[0]: item for item in overrides}
    for item in extra:
        merged[item.partition("=")[0]] = item
    return list(merged.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", required=True, choices=[*TIERS, "all"])
    parser.add_argument("--device", help="cpu, cuda or cuda:N (default: cpu for sweep60, cuda otherwise)")
    parser.add_argument("--seeds", type=parse_seeds, help="e.g. 0-4 or 0,3 (default: the tier's seeds)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args, extra = parser.parse_known_args()

    names = list(TIERS) if args.tier == "all" else [args.tier]
    for name in names:
        device = args.device or TIERS[name]["device"]
        if device == "cuda" and not args.device and not cuda_available():
            print(f"# {name}: torch finds no CUDA device, running on the CPU instead")
            device = "cpu"
        overrides = tier_overrides(name, device, args.seeds, extra)
        merged_device = next(o.partition("=")[2] for o in overrides if o.startswith("device="))
        env = dict(os.environ)
        if merged_device == "cpu":
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("MKL_NUM_THREADS", "2")
        sizes = {key: len(value.split(",")) for key, _, value in (o.partition("=") for o in overrides)}
        n_runs = sizes["model"] * sizes["dataset.delay"] * sizes["seed"]
        print(f"# {name}: {sizes['model']} models x {sizes['dataset.delay']} delays x "
              f"{sizes['seed']} seeds = {n_runs} runs")
        threads = (f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']} MKL_NUM_THREADS={env['MKL_NUM_THREADS']} "
                   if merged_device == "cpu" else "")
        print(threads + "sqs-bench -m " + shlex.join(overrides), flush=True)
        if args.dry_run:
            continue
        done = subprocess.run([sys.executable, "-m", "sqs_bench.cli", "-m", *overrides], env=env)
        if done.returncode != 0:
            print(f"tier {name} failed with exit code {done.returncode}", file=sys.stderr)
            return done.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
