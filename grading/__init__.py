"""Vendored grading harness: dataset adapters and the process-isolated scorer.

grading.datasets adapts each benchmark into Task objects; grading.evaluators grades
a candidate program against a benchmark's own tests in an isolated subprocess. Copied
from the self-correction-2026 repo (originally the "mend" framework, MIT) so seed
grading matches that experiment.
"""
