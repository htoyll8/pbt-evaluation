"""Mutation testing for PBT discrimination.

A PBT's real worth is whether it catches *wrong* programs, not whether the canonical
solution survives it (that is validity). We measure this by mutating the canonical solution
and checking how many mutants the PBT kills.

The interesting mutants are the ones that still pass the benchmark's unit tests: those are
wrong-but-unit-passing programs, i.e. synthetic overfit. A PBT's mutation score over that
set is a synthetic overfit-catch rate.

Mutants are single-point AST edits of the reference (swap a comparison/arithmetic/boolean
operator, bump an int constant, flip a bool). Equivalent mutants (behaviorally identical to
the reference) cannot be killed by any sound PBT; we do not detect them here, so a raw score
is a lower bound (a standard mutation-testing caveat).
"""
import ast

# Single-operator swaps applied one at a time to produce a mutant.
_SWAP = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt,
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
    ast.And: ast.Or, ast.Or: ast.And,
    ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd,  # covers set & / | and bitwise
}


class _Mutator(ast.NodeTransformer):
    """Mutate exactly the `target`-th eligible point; count points when target is -1."""

    def __init__(self, target: int):
        self.target = target
        self.n = -1

    def _op(self, op):
        if type(op) in _SWAP:
            self.n += 1
            if self.n == self.target:
                return _SWAP[type(op)]()
        return op

    def visit_Compare(self, node):
        self.generic_visit(node)
        node.ops = [self._op(o) for o in node.ops]
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        node.op = self._op(node.op)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        node.op = self._op(node.op)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, bool):           # bool before int (bool is an int subclass)
            self.n += 1
            if self.n == self.target:
                node.value = not node.value
        elif isinstance(node.value, int):
            self.n += 1
            if self.n == self.target:
                node.value = node.value + 1
        return node


def generate_mutants(code: str, limit: int = 20) -> list[str]:
    """Return up to `limit` single-point AST mutants of `code` (dedup, skip no-ops)."""
    try:
        counter = _Mutator(-1)
        counter.visit(ast.parse(code))
    except SyntaxError:
        return []
    total = counter.n + 1
    mutants, seen = [], {code.strip()}
    for t in range(total):
        tree = ast.parse(code)
        _Mutator(t).visit(tree)
        ast.fix_missing_locations(tree)
        try:
            src = ast.unparse(tree)
        except Exception:
            continue
        if src.strip() not in seen:
            seen.add(src.strip())
            mutants.append(src)
        if len(mutants) >= limit:
            break
    return mutants


def mutation_score(pbt_code: str, mutants: list[str], entry_point: str = "",
                   per_timeout: int = 5) -> dict:
    """Run the PBT against each mutant; a mutant is killed when the PBT fails on it.

    Returns {"killed": int, "total": int, "score": float|None}. `mutants` should already be
    filtered to the set you care about (e.g. those that pass the unit tests).
    """
    from pbt.core.score import run_property
    killed = 0
    for m in mutants:
        if not run_property(m, pbt_code, entry_point, per_timeout):  # PBT failed => caught
            killed += 1
    total = len(mutants)
    return {"killed": killed, "total": total, "score": (killed / total) if total else None}
