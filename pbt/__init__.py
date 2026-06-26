"""Cross-model property-based testing harness, SQLite-backed.

Rebuilds the JSONL pipeline around one primitive, score(program, suite), and a
normalized, append-only SQLite store keyed by content-hash IDs. Module layout:

    pbt.ids       content-hash IDs (dedup + cache keys)
    pbt.db        schema (DDL) + rw/ro connection helpers
    pbt.core      score(program, suite) + the Task/Program/Suite/Result types
    pbt.generate  produce programs and PBTs (artifacts; never grade)
    pbt.evaluate  score every program against every suite -> results
    pbt.analyze   the overfit metric + tables (read-only)
    pbt.run       config-driven orchestrator
"""