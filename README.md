# sqs-qlif-recall

Code, results and scripts behind the abstract *How to Recall Using Stochastic Quantum Spiking
Networks* (D. Brand, J. Chen, B. Rajendran, O. Simeone and F. Petruccione), an abstract for the
International Neuromorphic Computing Conference (INCC 2027).

The study puts two quantum spiking neurons on a controlled recall task. The stochastic quantum
spiking (SQS) neuron of Chen, Rajendran and Simeone (arXiv:2506.21324) measures its input-output
qubit at every step and keeps a memory qubit that is never measured. The quantum leaky
integrate-and-fire (QLIF) neuron of Brand and Petruccione (npj Quantum Inf. 10, 125 (2024); code at
https://github.com/deanbrand/QSNN) uses one qubit as both its memory and its measured output. In the
task, one of four cues appears for one step, then come *d* steps of random noise spikes and a go
signal, and the network must name the cue during a three-step response window; chance is 25%. The
published SQS circuit stays at chance at every delay from 1 to 16. Its two-qubit gates write to the
memory qubit, but none reads from it, so the memory state cannot change the spike probabilities.
Adding one *read gate* per neuron, a controlled rotation with the memory qubit as control and the
input-output qubit as target, lets the network recall the cue: 69% at delay 1 and still above chance
at delay 16. QLIF needs no read gate, because the qubit it measures is its memory, and it stays
within 5 points of SQS with the read gate at every delay.

Every network has 5 input lines (four cues and the go line), 64 hidden neurons and 4 output
neurons. Training uses Adam (learning rate 5e-4, batch 128) with surrogate gradients on a
cross-entropy loss over each output neuron's spike probability (SQS) or membrane (QLIF), summed over
the three response steps; accuracy reads the same quantity. Each seed draws 8000 training and 2000
test sequences.

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

This installs torch, numpy, hydra-core, omegaconf and matplotlib, plus pytest, scipy and PennyLane
for the tests. On Linux and Windows torch 2.11.0 comes from the PyTorch CUDA 12.8 index, the
version behind the recorded runs; with its CUDA libraries the environment takes about 4.6 GB, even
on a machine without a GPU. On macOS torch comes from PyPI, at a newer version (2.15.0 in
`uv.lock`), whose only macOS wheel is for Apple Silicon (arm64) on macOS 14 or later, so `uv sync`
fails on Intel Macs and on older macOS versions.

A GPU is optional: every command below runs on the CPU, `device=auto` (the default) uses CUDA
when torch finds it, and `run_experiments.py` moves its CUDA sets to the CPU when there is no GPU.

## Check the code

```
uv run pytest
```

309 tests, about a minute on the CPU, no network access. They compare the QLIF code with the
published file (`tests/reference/qsnn_alpha.py`), check the SQS statevector simulator against
PennyLane and against finite differences, train short runs of every model, and run the scripts
below on the shipped tables and on parts of them.

## Check that the published circuit cannot read its memory

```
uv run python scripts/check_memory_blindness.py
```

For one input-output and one memory qubit, the script builds 512 random neurons, puts each
neuron's memory qubit in 64 random states, applies one time step to the same input and prints
the largest change this makes to any spike probability. The published circuit (`crx_all`) gives
about 1e-15, floating-point error; the circuit with the read gate (`crx_both`) gives 0.12 to
0.97; the read gate with its angles fixed at 0 is back at 1e-15. This is the "to machine
precision" claim of the abstract. The script exits with status 1 if any of the three circuits
misbehaves.

## Reproduce the numbers and the figure

`results/` holds the 234 runs behind the abstract:

- `results/runs.csv`: one row per run (model, variant, delay, seed, `lead_in`, `dedupe_test`,
  sequence length `T`, epoch cap, epoch reached, why training stopped, final test accuracy and
  losses, trainable parameters, device, wall time). Accuracies are fractions in [0, 1].
- `results/test_history.csv`: test accuracy and loss after every epoch of every run.

```
uv run python scripts/report_numbers.py
uv run python scripts/plot_recall.py
```

`report_numbers.py` recomputes the 18 numeric claims of the abstract from the two tables and
prints, for each, the quoted value, the computed value and `ok`. It exits with status 1 if any
claim differs. Two claims need a definition the abstract leaves out. The read-gate gains of 29-33
points (delay 1) and 19-21 points (delay 2) are the means of two batches of five seeds, 0-4 and
5-9, each averaged over both initial angle choices; over all ten seeds the gains are 30.8 and
20.0 points. For when accuracy levels off, a run's final value is the mean test accuracy of its
last 10 epochs, and the run has levelled off at the first epoch whose mean over the 10 epochs up
to it is within 2 points of that.

`plot_recall.py` draws Figure 2(b), recall accuracy against delay, into
`figures/recall_vs_delay.png` and `.pdf`. It uses Times New Roman when installed and falls back
to another serif font. `figures/` is not tracked by git.

## Re-run the experiments

### One run

`sqs-bench` trains one (model, delay, seed) and writes one run directory. Settings are Hydra
overrides:

```
uv run sqs-bench model=qlif dataset.delay=1 train.epochs=10 variant=demo_delay1 device=cpu
```

This takes about ten seconds and reaches roughly 80% test accuracy. The accuracy the study
reports is `acc_mem` in the log and `test_accuracy_mem` in the final `done` line; it reads each
output neuron's spike probability (SQS) or membrane (QLIF). `test_accuracy` in the same line
counts output spikes instead, and is lower.

The run lands in `outputs/sqs_ce/memprobe/qlif_demo_delay1/seed0/` with `config.yaml` (every setting
used), `history.csv` (one row per epoch and split), `metrics.json` (the final record) and `run.log`.
`variant` names the run family and becomes part of the directory name; `outputs/` is not tracked by
git. SQS runs with the read gate first pass 30% after 5 to 240 epochs (fewer with uniform initial
angles, more at long delays), so a short SQS run ends near 25% and its `done` line ends with a
warning such as `warning: silent_readout` or `warning: at_uniform_loss` (the output neurons never
fire, or the loss is still that of a uniform guess). This is expected before training has converged.
`uv run sqs-bench --help` lists the models and every setting; `uv run sqs-bench --cfg job` prints
the composed settings without training.

### The five experiment sets

`scripts/run_experiments.py` launches the runs behind `results/runs.csv`, one experiment set at
a time, as one Hydra multirun each. The scripts and the `tier` column of `results/runs.csv` call
a set a *tier*. Finished runs are skipped, so an interrupted set can be started again with the
same command.

```
uv run python scripts/run_experiments.py --tier sweep60 --dry-run
uv run python scripts/run_experiments.py --tier plateau --seeds 0-1 --device cuda:0
uv run python scripts/run_experiments.py --tier all
uv run python scripts/run_experiments.py --tier all train.epochs=1 out_root=outputs/quick
```

`--dry-run` prints the `sqs-bench` command of each set and runs nothing. Anything after the
options is passed on as a Hydra override, for example `train.epochs=1` for a quick check, as in
the last line. A run counts as finished as soon as its `metrics.json` exists, whatever its
settings, so always give a quick check its own `out_root`: one-epoch runs left in `outputs/`
would be skipped by the real set and then collected as if they were real. Each run's
`config.yaml` holds its resolved settings, including the variant tag; `.hydra/overrides.yaml`
holds the overrides as passed, so there the tag reads `variant=<set>_delay${dataset.delay}`.

| Set (`--tier`) | What it tests | Models | Delays | Seeds | Epochs | Runs |
|---|---|---|---|---|---|---|
| `sweep60` | QLIF against the published SQS circuit over delay; the QLIF points of the figure | `qlif`, `sqsnn` | 1, 2, 4, 8, 16 | 0-2 | 60 | 30 |
| `long150` | The read gate over delay; the filled SQS points of the figure | `sqsnn`, `sqsnn_read` | 1, 2, 4, 8 | 0-4 | 150 | 40 |
| `plateau` | Longer training with a test set that has no sequence in common with the training set; the open points | `sqsnn`, `sqsnn_read` | 8, 16 | 0-4 | up to 300; stops when none of the last 60 epochs beats the best train loss before them by more than 1e-4 | 20 |
| `factorial` | Read gate on or off, crossed with small or uniform initial angles; the 29-33 and 19-21 point gains (means over seeds 0-4 and over seeds 5-9) | `sqsnn`, `sqsnn_read`, `sqsnn_uni`, `sqsnn_read_uni` | 1, 2 | 0-9 | 60 | 80 |
| `angle` | Read-gate angles frozen at their initial draw from U(0, a), a = 0, pi/4, pi/2, pi; same test set rule as `plateau` | `sqsnn_angle0`, `sqsnn_angle_pi4`, `sqsnn_angle_pi2`, `sqsnn_angle_pi` | 1, 2 | 0-7 | 60 | 64 |

All sets share `protocol=sqs_ce dataset=memprobe`. `sweep60` puts the cue on the first step
(`dataset.lead_in=0`) and runs on the CPU (`device=cpu`); the other four put one noise step
before the cue (`dataset.lead_in=1`) and run on CUDA. The QLIF comparison therefore crosses
sets: QLIF comes from `sweep60`, SQS with the read gate from `long150` and `plateau`.

### Compute

Wall time per run as recorded, with three runs sharing one device, and the total if the runs go
one after another:

| Set | Device | Per run | Total |
|---|---|---|---|
| `sweep60` | CPU | `qlif` 1.5-5 min; `sqsnn` 6.5-23 min, growing with the delay | 3.7 h |
| `long150` | GPU | 11-26 min | 10.8 h |
| `plateau` | GPU | `sqsnn` 13-36 min; `sqsnn_read` 32-80 min | 12.9 h |
| `factorial` | GPU | 4-6 min | 6.5 h |
| `angle` | GPU | 5-6 min | 5.7 h |
| all | | | about 40 h |

The model is small (a 4-amplitude statevector per neuron, 64 hidden neurons), so a GPU gives
little speed-up over a CPU: `sqsnn` at T = 6 took 6.5 s per epoch on the CPU against 4.5 s per
epoch on the GPU at T = 7. `run_experiments.py` runs one run at a time. To use several devices,
start one command per device with `--device cuda:0`, `--device cuda:1`, ... and a different
`--seeds` for each; all write into the same `outputs/`. On the CPU the script gives each run
two threads (`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`), as the recorded `sweep60` runs had; set
either variable yourself to use more.

Runs are deterministic: the same command on the same hardware gives the same numbers every time. The
recorded GPU runs used an NVIDIA V100 with torch 2.11.0 (CUDA 12.6) on Linux, and the recorded
`sweep60` runs used two CPU threads. On other hardware the first epochs match and the runs then
drift apart through rounding, because every spike is a random draw from a probability, and the final
accuracies differ by about the seed-to-seed spread. For example, one `sqsnn_angle_pi2` run that
recorded 46.5% ended at 45.2% on a different GPU, against a spread of 4 points over seeds. A `qlif`
run from `sweep60` reproduced its recorded accuracies exactly on a different CPU; only the loss
differed, by 1e-7. So `report_numbers.py` on a re-run from other hardware may print `DIFF` for some
claims, by amounts within the seed-to-seed spread.

### Collect and report

```
uv run python scripts/collect_results.py --outputs outputs --results my_results
uv run python scripts/report_numbers.py --results my_results
uv run python scripts/plot_recall.py --runs my_results/runs.csv --out my_figures
```

`collect_results.py` reads every finished run under `outputs/sqs_ce/memprobe/` whose variant is
`<set>_delay<d>` and writes the two tables (runs with other variants, such as the `demo_delay1`
run above, are left out); without arguments it writes into `results/`, over the shipped tables.
`collect_results.py` stops with an error if two runs share a model, variant and seed (for example
a re-run under another `train.run_suffix`), and leaves out, with a warning, runs that failed: the
loss became non-finite, or the final test accuracy or a loss is missing. `report_numbers.py`
checks a claim only when every set it draws on has all its seeds (`sweep60` 0-2, `long150` and
`plateau` 0-4, `factorial` 0-9, `angle` 0-7); otherwise it skips the claim and names the missing
seeds, and it does the same for empty or non-numeric cells, so a partial re-run gives skipped
claims rather than `DIFF`. `plot_recall.py` needs all of `sweep60`, `long150` and `plateau`, and
names the cells it is missing.

## Models

Every SQS model has one input-output qubit and one memory qubit per neuron, draws its spikes as
single shots from the measured probabilities, and uses the sigmoid surrogate gradient of the SQS
paper. Small initial angles are N(0, 0.1); uniform ones are U(0, pi).

| Model | Circuit | Initial angles | Parameters |
|---|---|---|---|
| `sqsnn` | published circuit (`crx_all`): controlled rotations from the input-output qubit onto the memory qubit, then a rotation on each qubit | small | 780 |
| `sqsnn_uni` | published circuit | uniform | 780 |
| `sqsnn_read` | published circuit plus the read gate (`crx_both`): a controlled rotation from the memory qubit onto the input-output qubit before the final rotations | small | 848 |
| `sqsnn_read_uni` | with the read gate | uniform | 848 |
| `sqsnn_angle0`, `sqsnn_angle_pi4`, `sqsnn_angle_pi2`, `sqsnn_angle_pi` | with the read gate; its angles are drawn from U(0, a) with a = 0, pi/4, pi/2 or pi and then frozen, every other parameter starts from the same values as `sqsnn_read_uni` at the same seed | uniform | 780 trainable, 848 stored |
| `qlif` | QLIF as published: one qubit per neuron, input and leak rotations, threshold 0.75, T1 = 60, S = 200; the gradient does not pass through emitted spikes or through time, so only the output layer learns | | 1288 |

At a = 0 the read gate is the identity and the circuit is the published one; this is where the
abstract's 24.8-24.9% comes from.

## Repository layout

```
src/sqs_bench/
  cli.py                 the sqs-bench command (Hydra)
  config.py              resolves the composed settings, tags run families
  conf/                  settings: dataset/memprobe, protocol/sqs_ce, train/surrogate, model/*
  data/memprobe.py       the recall task: cue, noise, go signal, response window
  neurons/sqs.py         SQS neuron: batched statevector simulation, both circuits, surrogate
  neurons/qlif.py        QLIF neuron, with the published Alpha / AlphaSurrogate code verbatim
  models/                feedforward spiking network built from one neuron type
  train/                 data loading, the training loop, early stopping
  losses.py              the cross-entropy objective over the response window
  metrics.py             decoders, accuracy, readout health checks
  utils/                 seeding, device choice, run directories
scripts/
  run_experiments.py     launch the five experiment sets
  collect_results.py     outputs/ -> results/runs.csv, results/test_history.csv
  report_numbers.py      results/ -> every number in the abstract
  plot_recall.py         results/runs.csv -> figures/recall_vs_delay.{png,pdf}
  check_memory_blindness.py
results/                 the 234 runs behind the abstract
tests/                   pytest suite; tests/reference/qsnn_alpha.py is the published QLIF file
```

## How to cite

The abstract:

> D. Brand, J. Chen, B. Rajendran, O. Simeone and F. Petruccione, "How to Recall Using
> Stochastic Quantum Spiking Networks", International Neuromorphic Computing Conference
> (INCC 2027).

The two neurons:

> J. Chen, B. Rajendran and O. Simeone, "Stochastic Quantum Spiking Neural Networks with Quantum
> Memory and Local Learning", IEEE J. Sel. Areas Commun. (2026), arXiv:2506.21324.

> D. Brand and F. Petruccione, "A quantum leaky integrate-and-fire spiking neuron and network",
> npj Quantum Inf. 10, 125 (2024).
