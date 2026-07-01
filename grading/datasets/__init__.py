"""Dataset registry.

Add a benchmark by writing a loader module that exposes `load(n_tasks) -> list[Task]`
and registering it in LOADERS below. Nothing else in the framework needs to change.
"""
from grading.datasets.base import Task
from grading.datasets import mbppplus, humaneval, apps

LOADERS = {
    "mbppplus": mbppplus.load,
    "humaneval": humaneval.load,
    "apps": apps.load,
}


def load_tasks(name: str, n_tasks: int, difficulties: tuple[str, ...] | None = None) -> list[Task]:
    """Load up to `n_tasks` of a registered benchmark as normalized Tasks. `difficulties`
    filters by tier and is only supported by datasets that carry one (currently apps)."""
    if name not in LOADERS:
        raise ValueError(f"dataset {name!r} not supported; choose from {sorted(LOADERS)}")
    if difficulties:
        if name != "apps":
            raise ValueError(f"difficulties filter is only supported for 'apps', not {name!r}")
        return LOADERS[name](n_tasks, difficulties=difficulties)
    return LOADERS[name](n_tasks)
