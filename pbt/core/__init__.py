"""The experiment's irreducible vocabulary and its one operation.

Types (Task / Program / Suite / Result) mirror the DB rows; score(program, suite) is the single graded operation that everything else reduces to.
"""
from pbt.core.score import run_property, score
from pbt.core.types import Program, Result, Suite, Task

__all__ = ["Task", "Program", "Suite", "Result", "score", "run_property"]
