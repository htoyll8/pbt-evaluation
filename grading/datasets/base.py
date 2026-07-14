"""
Benchmark-agnostic task representation shared by every dataset loader.
A loader turns a raw benchmark example into a `Task`. The evaluator scores a candidate by running `setup`, then the candidate program, then `prelude` (e.g. binding `candidate` for HumanEval), then each statement in `tests` (one assert each, for partial credit).

Weak vs strong suites:
    `tests` is the benchmark's ORIGINAL (weak) suite, and is what gates eligibility in
    pbt.analyze. `private_tests` is the strengthened (augmented) suite, when the benchmark
    ships one, and is seeded separately as a kind="private" suite. Keeping the two apart
    matters: gating eligibility on the strong suite silently drops every program that is
    overfit to the original benchmark, which is the population the study is about.
"""
from dataclasses import dataclass, field

from datasets import Dataset, load_dataset


@dataclass
class Task:
    """One benchmark problem, normalized across datasets."""
    task_id: str
    description: str          # the natural-language prompt shown to the model
    setup: str               # code executed before the candidate program
    tests: list              # ORIGINAL (weak) test units, each scored separately (=> partial
                             # credit). function mode: assert strings; stdio: [input, output]
    prelude: str = ""        # code executed after the program (e.g. bind `candidate`)
    per_timeout: int = 5     # seconds allowed per test unit (raise for whole-harness tests)
    io_mode: str = "function"  # "function" (assert-based) or "stdio" (piped stdin/stdout)
    difficulty: str = "na"   # benchmark difficulty tier, when the dataset provides one
    entry_point: str = ""    # name of the task's candidate function, when the dataset names one
    reference_solution: str = ""  # full runnable canonical solution (for the PBT validity filter)

    # The strengthened (augmented) suite, when the benchmark ships one. Seeded as a
    # kind="private" suite, never as the eligibility gate. Empty => the benchmark has none.
    private_tests: list = field(default_factory=list)
    # Execution context for `private_tests` when it differs from the task's own. None means
    # "reuse the task's value". Carried into the private suite blob by pbt.seed.
    private_prelude: str | None = None
    private_timeout: int | None = None

    @property
    def n_tests(self) -> int:
        return len(self.tests)


def take(source: str, n_tasks: int, split: str = "test") -> Dataset:
    """Load the first `n_tasks` rows of a HF dataset split, lazily.

    `ds.select` avoids materializing the whole split into a Python list before slicing,
    which matters for large benchmarks (e.g. APPS).
    """
    ds = load_dataset(source)[split]
    return ds.select(range(min(n_tasks, ds.num_rows)))
