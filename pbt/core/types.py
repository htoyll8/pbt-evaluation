"""The record types, mirroring the five tables.

Each carries its own content-hash ID (computed at construction via pbt.ids), so an object knows its identity before it ever touches the DB. Dedup and cache lookups then work the same in memory and on disk.
"""
from dataclasses import dataclass, field

from pbt import ids


@dataclass(frozen=True)
class Task:
    """One benchmark problem, normalized across datasets. Mirrors the `tasks` row.

    Attributes:
        task_id: Stable identifier for the problem.
        dataset: Name of the source benchmark (e.g. "mbppplus", "humaneval").
        prompt: Natural-language spec shown to the model.
        reference_solution: Canonical solution.
        setup: Code run before a candidate program.
        prelude: Code run after the program (e.g. bind `candidate` for HumanEval).
        per_timeout: Seconds allowed per test unit.
        io_mode: "function" (assert-based) or "stdio" (piped stdin/stdout).
        difficulty: Benchmark difficulty tier, when the dataset provides one.
    """
    task_id: str
    dataset: str
    prompt: str
    reference_solution: str = ""
    setup: str = ""
    prelude: str = ""
    per_timeout: int = 5
    io_mode: str = "function"
    difficulty: str = "na"


@dataclass(frozen=True)
class Program:
    """A deduped candidate solution. Mirrors the `programs` row.

    Attributes:
        task_id: ID of the task this program solves.
        prog_model: Model that produced the code.
        code: The program source.
        program_id: Content-hash ID; computed from the other fields if left blank.
    """
    task_id: str
    prog_model: str
    code: str
    program_id: str = field(default="", compare=False)

    def __post_init__(self):
        if not self.program_id:
            object.__setattr__(self, "program_id",
                               ids.program_id(self.task_id, self.prog_model, self.code))


@dataclass(frozen=True)
class Suite:
    """A deduped test of some `kind`. Mirrors the `suites` row.

    Attributes:
        task_id: ID of the task this suite tests.
        suite_model: Model that authored the suite.
        kind: One of "unit", "pbt", or "private".
        code: The test source.
        valid: None until checked; True/False once the reference solution has been run against it (the PBT validity filter).
        suite_id: Content-hash ID; computed from task_id/suite_model/code if left blank.

    Raises:
        ValueError: If kind is not one of "unit", "pbt", or "private".
    """
    task_id: str
    suite_model: str
    kind: str
    code: str
    valid: bool | None = None
    suite_id: str = field(default="", compare=False)

    def __post_init__(self):
        if self.kind not in ("unit", "pbt", "private"):
            raise ValueError(f"bad suite kind: {self.kind!r}")
        if not self.suite_id:
            object.__setattr__(self, "suite_id",
                               ids.suite_id(self.task_id, self.suite_model, self.code))


@dataclass(frozen=True)
class Result:
    """The outcome of score(program, suite). Mirrors the `results` row.

    Attributes:
        program_id: ID of the scored program.
        suite_id: ID of the suite it was scored against.
        task_id: ID of the task both share.
        prog_model: Model that wrote the program.
        suite_model: Model that wrote the suite.
        kind: Suite kind ("unit", "pbt", or "private").
        passed: Boolean verdict. For asserts this means pass_fraction == 1.0; for a PBT it means no counterexample was found.
        pass_fraction: Graded fraction in [0, 1] for partial credit.
    """
    program_id: str
    suite_id: str
    task_id: str
    prog_model: str
    suite_model: str
    kind: str
    passed: bool
    pass_fraction: float
