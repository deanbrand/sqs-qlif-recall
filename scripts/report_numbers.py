"""Recompute the numbers of the abstract, and the results behind it, from results/runs.csv and
results/test_history.csv.

    uv run python scripts/report_numbers.py
    uv run python scripts/report_numbers.py --results my_results

For each claim the script prints the value quoted (in the abstract or in the README), the value
computed here, and ``ok`` when the two agree to the number of digits quoted. The exit status is 1 if any
claim does not match.

A claim is checked only when every cell it uses has all the seeds of its tier (sweep60 0-2,
long150 0-4, plateau 0-4, factorial 0-9, angle 0-7). Otherwise, as after a partial re-run, it is
listed as skipped with the missing seeds, and so is a claim that meets an empty or non-numeric
cell.

Conventions:

* accuracies are in percent, 100 x ``test_accuracy_mem`` (spike probability for SQS, membrane
  for QLIF and LIF); a *cell* is one (model, tier, delay) and its value is the mean over seeds;
* chance is 25% (four cues) and a uniform guess has loss 3 ln 4 (cross entropy summed over the
  three scored steps);
* the read-gate gains of the factorial tier ("29-33", "19-21") are the means of two batches of
  five seeds, 0-4 and 5-9, of the per-seed gain averaged over the two initialisations; the
  pooled ten-seed mean is printed alongside;
* a run's *final value* is its mean test accuracy over its last 10 epochs, and the run has
  *levelled off* at the first epoch k >= 10 where the mean test accuracy of epochs k-9..k is
  within 2 points of its final value.

The claim that the memory state does not change the spike probabilities of the published
circuit comes from a simulation, not from training runs: see scripts/check_memory_blindness.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

CHANCE = 25.0
UNIFORM_LOSS = 3 * math.log(4)
CLAIMS = []

#: tier -> (seeds, models, delays) of the runs behind the abstract
TIERS = {
    "sweep60": (range(3), ("lif", "qlif", "sqsnn"), (1, 2, 4, 8, 16)),
    "long150": (range(5), ("sqsnn", "sqsnn_read"), (1, 2, 4, 8)),
    "plateau": (range(5), ("sqsnn", "sqsnn_read"), (8, 16)),
    "factorial": (range(10), ("sqsnn", "sqsnn_read", "sqsnn_uni", "sqsnn_read_uni"), (1, 2)),
    "angle": (range(8), ("sqsnn_angle0", "sqsnn_angle_pi4", "sqsnn_angle_pi2", "sqsnn_angle_pi"), (1, 2)),
}
LEVEL_WINDOW = 10   # epochs averaged for the final value and for each rolling mean
LEVEL_POINTS = 2.0  # how close to the final value counts as levelled off


class Missing(Exception):
    """The runs a claim needs are not in the tables, or a cell it reads is empty."""


def seed_text(seeds) -> str:
    """``[0, 1, 2, 4]`` -> ``"0-2, 4"``."""
    seeds, parts = sorted(seeds), []
    while seeds:
        end = 0
        while end + 1 < len(seeds) and seeds[end + 1] == seeds[end] + 1:
            end += 1
        parts.append(str(seeds[0]) if end == 0 else f"{seeds[0]}-{seeds[end]}")
        seeds = seeds[end + 1:]
    return ", ".join(parts)


def parse(text) -> float:
    """A table cell as a number; NaN when it is empty or not a finite number."""
    try:
        value = float((text or "").strip())
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def number(row: dict, column: str) -> float:
    """The finite number in ``row[column]``; :class:`Missing` if the cell is empty or not finite."""
    value = parse(row.get(column))
    if math.isnan(value):
        raise Missing(f"{row.get('model')} {row.get('variant')} seed {row.get('seed')}: "
                      f"{column} is {row.get(column)!r}, not a finite number")
    return value


def claim(text):
    def register(fn):
        CLAIMS.append((text, fn))
        return fn
    return register


class Results:
    def __init__(self, folder: Path) -> None:
        with (folder / "runs.csv").open(newline="", encoding="utf-8") as fh:
            self.runs = list(csv.DictReader(fh))
        # (model, variant, seed) -> test accuracy per epoch, NaN where the cell is empty
        self.curves: dict[tuple[str, str, int], list[float]] = {}
        # cells with a history row whose seed or epoch is not a number: their curves are unusable
        self.bad_history: set[tuple[str, str]] = set()
        with (folder / "test_history.csv").open(newline="", encoding="utf-8") as fh:
            points = []
            for h in csv.DictReader(fh):
                seed, epoch = parse(h.get("seed")), parse(h.get("epoch"))
                if math.isnan(seed) or math.isnan(epoch):
                    self.bad_history.add((h.get("model"), h.get("variant")))
                    continue
                points.append((epoch, (h["model"], h["variant"], int(seed)), parse(h.get("test_acc_mem"))))
            for _, key, value in sorted(points, key=lambda p: p[0]):
                self.curves.setdefault(key, []).append(100 * value)

    def rows(self, model: str, variant: str) -> list[dict]:
        """The runs of one cell; :class:`Missing` unless every seed of its tier is there."""
        found = [r for r in self.runs if r["model"] == model and r["variant"] == variant]
        tier = variant.rsplit("_delay", 1)[0]
        needed = set(TIERS[tier][0])
        missing = sorted(needed - {int(number(r, "seed")) for r in found})
        if missing:
            have = {0: "no runs", 1: "1 run"}.get(len(found), f"{len(found)} runs")
            raise Missing(f"{model} {variant}: {have}, missing seeds {seed_text(missing)} "
                          f"of {tier} ({seed_text(needed)})")
        return found

    def require_tier(self, tier: str) -> None:
        """:class:`Missing` unless every cell of ``tier`` has all its seeds."""
        _, models, delays = TIERS[tier]
        for model in models:
            for delay in delays:
                self.rows(model, f"{tier}_delay{delay}")

    def acc(self, model: str, variant: str) -> dict[int, float]:
        """Seed -> final test accuracy in percent."""
        return {int(number(r, "seed")): 100 * number(r, "test_accuracy_mem")
                for r in self.rows(model, variant)}

    def mean(self, model: str, variant: str) -> float:
        return statistics.mean(self.acc(model, variant).values())

    def paired(self, model_a: str, model_b: str, variant: str) -> list[float]:
        """Per-seed difference a - b over the seeds both models ran."""
        a, b = self.acc(model_a, variant), self.acc(model_b, variant)
        common = sorted(a.keys() & b.keys())
        if not common:
            raise Missing(f"{model_a} and {model_b} share no seed in {variant}")
        return [a[s] - b[s] for s in common]

    def levelled(self, model: str, variant: str) -> list[int]:
        """Levelled-off epoch of every seed of a cell: the first epoch whose mean over the
        last LEVEL_WINDOW epochs is within LEVEL_POINTS of the final value, the mean of the
        run's last LEVEL_WINDOW epochs."""
        rows = self.rows(model, variant)
        if (model, variant) in self.bad_history:
            raise Missing(f"{model} {variant}: test_history.csv has rows whose seed or epoch "
                          "is not a number")
        out, w = [], LEVEL_WINDOW
        for row in rows:
            seed = int(number(row, "seed"))
            curve = self.curves.get((model, variant, seed))
            if curve is None:
                raise Missing(f"{model} {variant} seed {seed} has no rows in test_history.csv")
            if not all(map(math.isfinite, curve)):
                raise Missing(f"{model} {variant} seed {seed}: test_history.csv has an empty or "
                              "non-finite test_acc_mem")
            if len(curve) < w:
                raise Missing(f"{model} {variant} seed {seed} has fewer than {w} epochs")
            final = statistics.mean(curve[-w:])
            out.append(next(k for k in range(w, len(curve) + 1)
                            if abs(statistics.mean(curve[k - w:k]) - final) <= LEVEL_POINTS))
        return out


def nearest_ten(x: float) -> int:
    return 10 * math.floor(x / 10 + 0.5)


def span(values, fmt=".1f") -> str:
    """``"min-max"`` of ``values`` in format ``fmt``."""
    return f"{min(values):{fmt}}-{max(values):{fmt}}"


NO_READ_GATE_CELLS = (
    [f"sweep60_delay{d}" for d in (1, 2, 4, 8, 16)]
    + [f"long150_delay{d}" for d in (1, 2, 4, 8)]
    + [f"plateau_delay{d}" for d in (8, 16)]
)
READ_GATE_CELLS = [f"long150_delay{d}" for d in (1, 2, 4, 8)] + ["plateau_delay8", "plateau_delay16"]


# Each claim returns (value in the abstract, value computed here, match, detail).
@claim("Without the read gate SQS stays at chance at every delay from 1 to 16")
def _(res):
    means = [res.mean("sqsnn", v) for v in NO_READ_GATE_CELLS]
    return ("24.5-25.4%", f"{span(means, '.2f')}%", span(means) == "24.5-25.4",
            f"sqsnn cell means, {len(means)} cells: sweep60 d1-16, long150 d1-8, plateau d8 and d16")


@claim("... and its loss stays at the value of a uniform guess")
def _(res):
    rows = [r for v in NO_READ_GATE_CELLS for r in res.rows("sqsnn", v)]
    test = max(abs(number(r, "test_loss") - UNIFORM_LOSS) for r in rows)
    train = max(abs(number(r, "train_loss") - UNIFORM_LOSS) for r in rows)
    return (f"3 ln 4 = {UNIFORM_LOSS:.3f}", f"within {max(test, train):.4f} of it", max(test, train) < 0.005,
            f"largest |loss - 3 ln 4| over the {len(rows)} sqsnn runs: test {test:.4f}, train {train:.4f}")


@claim("... even with up to 300 epochs of training")
def _(res):
    rows = res.rows("sqsnn", "plateau_delay8") + res.rows("sqsnn", "plateau_delay16")
    cap = max(int(number(r, "epochs")) for r in rows)
    stopped = [int(number(r, "stopped_epoch")) for r in rows]
    by_rule = sum(r["stop_reason"] == "criterion" for r in rows)
    return ("300", str(cap), cap == 300,
            f"epoch cap of the sqsnn plateau runs; they stopped at epochs {span(stopped, 'd')}, "
            f"{by_rule} of {len(rows)} because the train loss levelled off")


@claim("With the read gate, accuracy reaches 69% at delay 1")
def _(res):
    acc = list(res.acc("sqsnn_read", "long150_delay1").values())
    m = statistics.mean(acc)
    spread = f"{statistics.stdev(acc):.2f} s.d. over {len(acc)} seeds" if len(acc) > 1 else "one seed"
    return ("69%", f"{m:.2f}%", f"{m:.0f}" == "69", f"sqsnn_read long150 delay 1, {spread}")


@claim("... and stays above chance at delay 16 (30%)")
def _(res):
    acc = list(res.acc("sqsnn_read", "plateau_delay16").values())
    m = statistics.mean(acc)
    above = sum(a > CHANCE for a in acc)
    return ("30%", f"{m:.2f}%", f"{m:.0f}" == "30" and above == len(acc),
            f"sqsnn_read plateau delay 16, {m - CHANCE:+.2f} points over chance; above chance in "
            f"{above} of {len(acc)} seeds ({span(acc, '.1f')}%)")


def read_gate_gain(res, delay: int) -> tuple[list[float], list[float], list[float]]:
    """Factorial tier: per-seed gain of the read gate for each initialisation, and its average."""
    variant = f"factorial_delay{delay}"
    small = res.paired("sqsnn_read", "sqsnn", variant)
    uniform = res.paired("sqsnn_read_uni", "sqsnn_uni", variant)
    return small, uniform, [(a + b) / 2 for a, b in zip(small, uniform)]


def batch_means(gain: list[float], variant: str) -> tuple[list[float], float]:
    """Mean gain over the batch of seeds 0-4, over the batch of seeds 5-9, and over all ten."""
    if len(gain) != 10:
        raise Missing(f"{variant}: needs 10 paired seeds for the two batches, found {len(gain)}")
    return [statistics.mean(gain[:5]), statistics.mean(gain[5:])], statistics.mean(gain)


BATCH_DETAIL = ("means of two batches of five seeds (0-4 and 5-9) of the per-seed gain, each "
                "averaged over the two initialisations; pooled ten-seed mean {pooled:.1f}")


@claim("10 seeds, 60 epochs: the read gate adds on average 29-33 points at delay 1")
def _(res):
    *_, gain = read_gate_gain(res, 1)
    means, pooled = batch_means(gain, "factorial_delay1")
    return ("29-33", " and ".join(f"{h:.2f}" for h in means), span(means, ".0f") == "29-33",
            BATCH_DETAIL.format(pooled=pooled))


@claim("... and 19-21 points at delay 2")
def _(res):
    *_, gain = read_gate_gain(res, 2)
    means, pooled = batch_means(gain, "factorial_delay2")
    return ("19-21", " and ".join(f"{h:.2f}" for h in means), span(means, ".0f") == "19-21",
            BATCH_DETAIL.format(pooled=pooled))


@claim("... and helped in every seed")
def _(res):
    pairs = []
    for d in (1, 2):
        small, uniform, _ = read_gate_gain(res, d)
        pairs += small + uniform
    rows =[r for m in ("sqsnn", "sqsnn_read", "sqsnn_uni", "sqsnn_read_uni")
            for d in (1, 2) for r in res.rows(m, f"factorial_delay{d}")]
    seeds = {r["seed"] for r in rows}
    epochs = {r["epochs"] for r in rows}
    positive = sum(g > 0 for g in pairs)
    return ("every seed", f"{positive} of {len(pairs)}", positive == len(pairs) == 40 and len(seeds) == 10,
            f"paired gains, 2 delays x 2 initialisations x {len(seeds)} seeds; smallest {min(pairs):.2f}; "
            f"epochs {','.join(sorted(epochs))}")


@claim("Read-gate angle 0 (the published circuit) scores 24.8-24.9%")
def _(res):
    means = [res.mean("sqsnn_angle0", f"angle_delay{d}") for d in (1, 2)]
    return ("24.8-24.9%", f"{means[0]:.2f}% and {means[1]:.2f}%", span(means) == "24.8-24.9",
            "sqsnn_angle0 at delays 1 and 2")


@claim("Angle pi/2 adds 26.7 points at delay 1 in 8 of 8 seeds")
def _(res):
    gain = res.paired("sqsnn_angle_pi2", "sqsnn_angle0", "angle_delay1")
    m, n_pos = statistics.mean(gain), sum(g > 0 for g in gain)
    return ("+26.7, 8 of 8", f"+{m:.2f}, {n_pos} of {len(gain)}", f"{m:.1f}" == "26.7" and n_pos == len(gain) == 8,
            "per-seed sqsnn_angle_pi2 minus sqsnn_angle0")


@claim("Angle pi scores below pi/2 on average")
def _(res):
    parts, means = [], []
    for d in (1, 2):
        diff = res.paired("sqsnn_angle_pi", "sqsnn_angle_pi2", f"angle_delay{d}")
        means.append(statistics.mean(diff))
        parts.append(f"delay {d}: lower in {sum(x < 0 for x in diff)} of {len(diff)} seeds")
    return ("below", " and ".join(f"{m:+.2f}" for m in means), all(m < 0 for m in means),
            "mean per-seed sqsnn_angle_pi minus sqsnn_angle_pi2 at delays 1 and 2; " + ", ".join(parts))


@claim("QLIF is close to SQS with the read gate: within 5 points at every delay")
def _(res):
    pairs = [(d, f"long150_delay{d}") for d in (1, 2, 4, 8)] + [(8, "plateau_delay8"), (16, "plateau_delay16")]
    diffs = [(d, v, res.mean("qlif", f"sweep60_delay{d}") - res.mean("sqsnn_read", v)) for d, v in pairs]
    worst = max(abs(x) for *_, x in diffs)
    return ("within 5", f"at most {worst:.2f} apart", worst < 5,
            "qlif minus sqsnn_read: " + ", ".join(f"d{d} {x:+.2f} ({v.split('_')[0]})" for d, v, x in diffs))


@claim("QLIF: 72% at delay 1, 35% at delay 8")
def _(res):
    d1, d8 = res.mean("qlif", "sweep60_delay1"), res.mean("qlif", "sweep60_delay8")
    return ("72% and 35%", f"{d1:.2f}% and {d8:.2f}%", (f"{d1:.0f}", f"{d8:.0f}") == ("72", "35"),
            "qlif sweep60 cell means")


@claim("LIF: 96% at delay 1, 75% at delay 8")
def _(res):
    d1, d8 = res.mean("lif", "sweep60_delay1"), res.mean("lif", "sweep60_delay8")
    return ("96% and 75%", f"{d1:.2f}% and {d8:.2f}%", (f"{d1:.0f}", f"{d8:.0f}") == ("96", "75"),
            "lif sweep60 cell means")


@claim("LIF, QLIF and SQS networks have the same width, 64 hidden neurons")
def _(res):
    params = {m: {int(number(r, "n_params_trainable")) for d in (1, 2, 4, 8, 16)
                  for r in res.rows(m, f"sweep60_delay{d}")} for m in ("lif", "qlif", "sqsnn")}
    # 5 inputs -> 64 hidden -> 4 outputs: LIF 9 x 64 weights, QLIF two planes with bias, SQS 780
    expected = {"lif": {(5 + 4) * 64}, "qlif": {2 * ((5 + 1) * 64 + (64 + 1) * 4)}, "sqsnn": {780}}
    return ("64", "64" if params == expected else "other", params == expected,
            "trainable parameters per sweep60 run: "
            + ", ".join(f"{m} {'/'.join(map(str, sorted(v)))}" for m, v in params.items()))


LEVEL_DETAIL = (f"final value = mean test accuracy of the last {LEVEL_WINDOW} epochs; levelled off "
                f"= first epoch whose {LEVEL_WINDOW}-epoch mean is within {LEVEL_POINTS:g} points of it")


@claim("QLIF accuracy levels off within about 50 of 60 epochs")
def _(res):
    epochs = [k for d in (1, 2, 4, 8, 16) for k in res.levelled("qlif", f"sweep60_delay{d}")]
    return ("about 50", f"{max(epochs)} (to the nearest 10: {nearest_ten(max(epochs))})",
            nearest_ten(max(epochs)) == 50,
            f"latest of {len(epochs)} qlif runs; median {statistics.median(epochs):g}, range {span(epochs, 'd')}; "
            + LEVEL_DETAIL)


@claim("SQS with the read gate levels off within 70-220 epochs")
def _(res):
    epochs = [k for v in READ_GATE_CELLS for k in res.levelled("sqsnn_read", v)]
    lo, hi = min(epochs), max(epochs)
    return ("70-220", f"{lo}-{hi} (to the nearest 10: {nearest_ten(lo)}-{nearest_ten(hi)})",
            (nearest_ten(lo), nearest_ten(hi)) == (70, 220),
            f"{len(epochs)} sqsnn_read runs: long150 d1-8 and plateau d8, d16; " + LEVEL_DETAIL)


@claim("At delay 8, runs of 150 and of up to 300 epochs agree")
def _(res):
    a, b = res.mean("sqsnn_read", "long150_delay8"), res.mean("sqsnn_read", "plateau_delay8")
    return ("39.7% and 40.1%", f"{a:.2f}% and {b:.2f}%", (f"{a:.1f}", f"{b:.1f}") == ("39.7", "40.1"),
            "sqsnn_read delay 8, long150 and plateau")


@claim("Figure 1(e): seeds and epochs per series")
def _(res):
    def series(models, tier, delays):
        """Seeds per cell and epoch caps of a series, each as "a" or "a/b" if they vary."""
        cells = [res.rows(m, f"{tier}_delay{d}") for m in models for d in delays]
        seeds = sorted({len(c) for c in cells})
        caps = sorted({int(number(r, "epochs")) for c in cells for r in c})
        return "/".join(map(str, seeds)), "/".join(map(str, caps))
    sqs = ("sqsnn", "sqsnn_read")
    (s1, e1), (s2, e2) = series(sqs, "long150", (1, 2, 4, 8)), series(sqs, "plateau", (8, 16))
    s3, e3 = series(["qlif", "lif"], "sweep60", (1, 2, 4, 8, 16))
    quoted = "SQS 5 seeds, 150 or <=300 epochs; QLIF and LIF 3 seeds, 60 epochs"
    got = (f"SQS {s1 if s1 == s2 else s1 + ' and ' + s2} seeds, {e1} or <={e2} epochs; "
           f"QLIF and LIF {s3} seeds, {e3} epochs")
    return (quoted, got, got == quoted, "SQS: long150 (filled) and plateau (open); QLIF and LIF: sweep60")


@claim("The open points and the angle test use test sets with no train overlap")
def _(res):
    flags = {}
    for tier in ("long150", "plateau", "angle"):
        res.require_tier(tier)
        flags[tier] = {r["dedupe_test"] for r in res.runs if r["tier"] == tier}
    ok = flags["plateau"] == flags["angle"] == {"true"} and flags["long150"] == {"false"}
    return ("plateau, angle", ", ".join(t for t, f in flags.items() if f == {"true"}), ok,
            "tiers whose every run has dedupe_test=true (long150, the filled points, has it off)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"), help="folder with the two tables")
    args = parser.parse_args()
    res = Results(args.results)
    print(f"{len(res.runs)} runs, {sum(map(len, res.curves.values()))} test epochs from {args.results}\n")

    counts = {"ok": 0, "DIFF": 0, "missing": 0}
    for text, fn in CLAIMS:
        try:
            quoted, computed, match, detail = fn(res)
        except (Missing, statistics.StatisticsError) as err:  # too few runs, e.g. a partial re-run
            counts["missing"] += 1
            print(f"  --  {text}\n      skipped: {err}\n")
            continue
        status = "ok" if match else "DIFF"
        counts[status] += 1
        print(f"{status:>4}  {text}\n      quoted: {quoted:<24} computed: {computed}\n      ({detail})\n")
    print(f"{counts['ok']} of {len(CLAIMS)} claims match, {counts['DIFF']} differ, "
          f"{counts['missing']} skipped.")
    print("Memory state vs spike probabilities: run scripts/check_memory_blindness.py")
    return 1 if counts["DIFF"] else 0


if __name__ == "__main__":
    sys.exit(main())
