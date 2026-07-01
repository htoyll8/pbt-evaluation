"""Content-hash IDs for dedup, caching, and idempotency.

An ID is a hash of the content that defines an artifact's identity. The same
(task, model, code) yields the same ID, so re-generation dedupes and score()
becomes a cache lookup keyed by (program_id, suite_id).

The model is part of the identity: two models can emit byte-identical code, butthe study asks which model wrote it, so identity is (task, model, content), not content alone. This also keeps the two models' work distinct.
"""
import hashlib

_LEN = 16  # 64 bits of sha256, hex; ample to avoid collisions at study scale (~1e4 rows)


def _hash(*components: str) -> str:
    """Compute a stable, truncated SHA-256 over the given identity components.

    Components are NUL-joined before hashing so that ('a', 'bc') and ('ab', 'c') map to different digests rather than colliding on the concatenation 'abc'.

    Args:
        *components: Strings whose ordered content defines the identity being hashed.

    Returns:
        The first 16 hex characters (64 bits) of the SHA-256 digest.
    """
    h = hashlib.sha256("\x00".join(components).encode("utf-8"))
    return h.hexdigest()[:_LEN]


def program_id(task_id: str, prog_model: str, code: str) -> str:
    """Return the content-hash ID for a candidate program.

    Args:
        task_id: ID of the task the program solves.
        prog_model: Name of the model that produced the code.
        code: The program source. Identical code from different models stays
            distinct, since the model is part of the identity.

    Returns:
        A stable ID; identical (task, model, code) always hashes the same.
    """
    return _hash("program", task_id, prog_model, code)


def suite_id(task_id: str, suite_model: str, code: str) -> str:
    """Return the content-hash ID for a test suite of any kind.

    Args:
        task_id: ID of the task the suite tests.
        suite_model: Name of the model that authored the suite.
        code: The suite source.

    Returns:
        A stable ID; identical (task, model, code) always hashes the same.
    """
    return _hash("suite", task_id, suite_model, code)


def gen_id(task_id: str, prog_model: str, sample_index: int) -> str:
    """Return the ID for one sampling event.

    Deterministic in (task, model, sample_index) so re-running the same sample is idempotent (an INSERT OR IGNORE no-ops) instead of duplicating events. The code is deliberately excluded: a generation's identity is the event, not its output, which lets the same program sampled n times stay one program.

    Args:
        task_id: ID of the task being sampled.
        prog_model: Name of the model doing the sampling.
        sample_index: Zero-based index of this draw within the task/model's samples.

    Returns:
        A stable ID for the (task, model, sample_index) event.
    """
    return _hash("gen", task_id, prog_model, str(sample_index))
